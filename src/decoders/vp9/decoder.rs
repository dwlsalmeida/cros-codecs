// Copyright 2022 The ChromiumOS Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

use anyhow::anyhow;
use anyhow::Result;
use log::debug;

use crate::decoders::vp9::backends::StatelessDecoderBackend;
use crate::decoders::vp9::lookups::AC_QLOOKUP;
use crate::decoders::vp9::lookups::AC_QLOOKUP_10;
use crate::decoders::vp9::lookups::AC_QLOOKUP_12;
use crate::decoders::vp9::lookups::DC_QLOOKUP;
use crate::decoders::vp9::lookups::DC_QLOOKUP_10;
use crate::decoders::vp9::lookups::DC_QLOOKUP_12;
use crate::decoders::vp9::parser::BitDepth;
use crate::decoders::vp9::parser::Frame;
use crate::decoders::vp9::parser::FrameType;
use crate::decoders::vp9::parser::Header;
use crate::decoders::vp9::parser::Parser;
use crate::decoders::vp9::parser::Profile;
use crate::decoders::vp9::parser::INTRA_FRAME;
use crate::decoders::vp9::parser::LAST_FRAME;
use crate::decoders::vp9::parser::MAX_LOOP_FILTER;
use crate::decoders::vp9::parser::MAX_MODE_LF_DELTAS;
use crate::decoders::vp9::parser::MAX_REF_FRAMES;
use crate::decoders::vp9::parser::MAX_SEGMENTS;
use crate::decoders::vp9::parser::NUM_REF_FRAMES;
use crate::decoders::vp9::parser::SEG_LVL_ALT_L;
use crate::decoders::vp9::parser::SEG_LVL_REF_FRAME;
use crate::decoders::vp9::parser::SEG_LVL_SKIP;
use crate::decoders::BlockingMode;
use crate::decoders::DecodedHandle;
use crate::decoders::Result as VideoDecoderResult;
use crate::decoders::VideoDecoder;
use crate::Resolution;

/// Represents where we are in the negotiation status. We assume ownership of
/// the incoming buffers in this special case so that clients do not have to do
/// the bookkeeping themselves.
enum NegotiationStatus {
    /// Still waiting for a key frame.
    NonNegotiated,

    /// Saw a key frame. Negotiation is possible until the next call to decode()
    Possible {
        key_frame: (u64, Box<Header>, Vec<u8>),
    },

    /// Negotiated. Locks in the format until a new key frame is seen if that
    /// new key frame changes the stream parameters.
    Negotiated,
}

impl Default for NegotiationStatus {
    fn default() -> Self {
        Self::NonNegotiated
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Segmentation {
    /// Loop filter level
    pub lvl_lookup: [[u8; MAX_MODE_LF_DELTAS]; MAX_REF_FRAMES],

    /// AC quant scale for luma component
    pub luma_ac_quant_scale: i16,
    /// DC quant scale for luma component
    pub luma_dc_quant_scale: i16,
    /// AC quant scale for chroma component
    pub chroma_ac_quant_scale: i16,
    /// DC quant scale for chroma component
    pub chroma_dc_quant_scale: i16,

    /// Whether the alternate reference frame segment feature is enabled (SEG_LVL_REF_FRAME)
    pub reference_frame_enabled: bool,
    /// The feature data for the reference frame featire
    pub reference_frame: i16,
    /// Whether the skip segment feature is enabled (SEG_LVL_SKIP)
    pub reference_skip_enabled: bool,
}

pub struct Decoder<T: DecodedHandle> {
    /// A parser to extract bitstream data and build frame data in turn
    parser: Parser,

    /// Whether the decoder should block on decode operations.
    blocking_mode: BlockingMode,

    /// The backend used for hardware acceleration.
    backend: Box<dyn StatelessDecoderBackend<Handle = T>>,

    /// Keeps track of whether the decoded format has been negotiated with the
    /// backend.
    negotiation_status: NegotiationStatus,

    /// The current resolution
    coded_resolution: Resolution,

    /// A queue with the pictures that are ready to be sent to the client.
    ready_queue: Vec<T>,

    /// A monotonically increasing counter used to tag pictures in display
    /// order
    current_display_order: u64,

    /// The reference frames in use.
    reference_frames: [Option<T>; NUM_REF_FRAMES],

    /// Per-segment data.
    segmentation: [Segmentation; MAX_SEGMENTS],

    /// Cached value for bit depth
    bit_depth: BitDepth,
    /// Cached value for profile
    profile: Profile,
}

impl<T: DecodedHandle + Clone + 'static> Decoder<T> {
    /// Create a new codec backend for VP8.
    #[cfg(any(feature = "vaapi", test))]
    pub(crate) fn new(
        backend: Box<dyn StatelessDecoderBackend<Handle = T>>,
        blocking_mode: BlockingMode,
    ) -> Result<Self> {
        Ok(Self {
            backend,
            blocking_mode,
            parser: Default::default(),
            negotiation_status: Default::default(),
            reference_frames: Default::default(),
            segmentation: Default::default(),
            coded_resolution: Default::default(),
            ready_queue: Default::default(),
            bit_depth: Default::default(),
            profile: Default::default(),
            current_display_order: Default::default(),
        })
    }

    /// Replace a reference frame with `handle`.
    fn replace_reference(reference: &mut Option<T>, handle: &T) {
        *reference = Some(handle.clone());
    }

    fn update_references(
        reference_frames: &mut [Option<T>; NUM_REF_FRAMES],
        picture: &T,
        mut refresh_frame_flags: u8,
    ) -> Result<()> {
        #[allow(clippy::needless_range_loop)]
        for i in 0..NUM_REF_FRAMES {
            if (refresh_frame_flags & 1) == 1 {
                debug!("Replacing reference frame {}", i);
                Self::replace_reference(&mut reference_frames[i], picture)
            }

            refresh_frame_flags >>= 1;
        }

        Ok(())
    }

    fn block_on_one(&mut self) -> Result<()> {
        if let Some(handle) = self.ready_queue.first() {
            return handle.sync().map_err(|e| e.into());
        }

        Ok(())
    }

    /// Returns the ready handles.
    fn get_ready_frames(&mut self) -> Vec<T> {
        // Count all ready handles.
        let num_ready = self
            .ready_queue
            .iter()
            .take_while(|&handle| self.backend.handle_is_ready(handle))
            .count();

        let retain = self.ready_queue.split_off(num_ready);
        // `split_off` works the opposite way of what we would like, leaving [0..num_ready) in
        // place, so we need to swap `retain` with `ready_queue`.
        let ready = std::mem::take(&mut self.ready_queue);
        self.ready_queue = retain;

        ready
    }

    /// Handle a single frame.
    fn handle_frame(&mut self, frame: &Frame<&[u8]>, timestamp: u64) -> Result<()> {
        let mut decoded_handle = if frame.header.show_existing_frame {
            // Frame to be shown. Unwrapping must produce a Picture, because the
            // spec mandates frame_to_show_map_idx references a valid entry in
            // the DPB
            let idx = usize::from(frame.header.frame_to_show_map_idx);
            let ref_frame = self.reference_frames[idx].as_ref().unwrap();

            // We are done, no further processing needed.
            ref_frame.clone()
        } else {
            // Otherwise, we must actually arrange to decode a frame
            let refresh_frame_flags = frame.header.refresh_frame_flags;

            let offset = frame.offset;
            let size = frame.size;

            let block = matches!(self.blocking_mode, BlockingMode::Blocking)
                || matches!(self.negotiation_status, NegotiationStatus::Possible { .. });

            let bitstream = &frame.bitstream[offset..offset + size];

            Self::update_segmentation(&frame.header, &mut self.segmentation)?;
            let decoded_handle = self
                .backend
                .submit_picture(
                    &frame.header,
                    &self.reference_frames,
                    bitstream,
                    timestamp,
                    &self.segmentation,
                )
                .map_err(|e| anyhow!(e))?;

            if block {
                decoded_handle.sync()?;
            }

            // Do DPB management
            Self::update_references(
                &mut self.reference_frames,
                &decoded_handle,
                refresh_frame_flags,
            )?;

            decoded_handle
        };

        let show_existing_frame = frame.header.show_existing_frame;
        if frame.header.show_frame || show_existing_frame {
            let order = self.current_display_order;
            decoded_handle.set_display_order(order);
            self.current_display_order += 1;

            self.ready_queue.push(decoded_handle);
        }

        Ok(())
    }

    fn negotiation_possible(&self, frame: &Frame<impl AsRef<[u8]>>) -> bool {
        let coded_resolution = self.coded_resolution;
        let hdr = &frame.header;
        let width = hdr.width;
        let height = hdr.height;
        let bit_depth = hdr.bit_depth;
        let profile = hdr.profile;

        if width == 0 || height == 0 {
            return false;
        }

        width != coded_resolution.width
            || height != coded_resolution.height
            || bit_depth != self.bit_depth
            || profile != self.profile
    }

    /// A clamp such that min <= x <= max
    fn clamp<U: PartialOrd>(x: U, low: U, high: U) -> U {
        if x > high {
            high
        } else if x < low {
            low
        } else {
            x
        }
    }

    /// An implementation of seg_feature_active as per "6.4.9 Segmentation feature active syntax"
    fn seg_feature_active(hdr: &Header, segment_id: u8, feature: u8) -> bool {
        let feature_enabled = hdr.seg.feature_enabled;
        hdr.seg.enabled && feature_enabled[usize::from(segment_id)][usize::from(feature)]
    }

    /// An implementation of get_qindex as per "8.6.1 Dequantization functions"
    fn get_qindex(hdr: &Header, segment_id: u8) -> i32 {
        let base_q_idx = hdr.quant.base_q_idx;

        if Self::seg_feature_active(hdr, segment_id, 0) {
            let mut data = hdr.seg.feature_data[usize::from(segment_id)][0] as i32;

            if !hdr.seg.abs_or_delta_update {
                data += i32::from(base_q_idx);
            }

            Self::clamp(data, 0, 255)
        } else {
            i32::from(base_q_idx)
        }
    }

    /// An implementation of get_dc_quant as per "8.6.1 Dequantization functions"
    fn get_dc_quant(hdr: &Header, segment_id: u8, luma: bool) -> Result<i32> {
        let delta_q_dc = if luma {
            hdr.quant.delta_q_y_dc
        } else {
            hdr.quant.delta_q_uv_dc
        };
        let qindex = Self::get_qindex(hdr, segment_id);
        let q_table_idx = Self::clamp(qindex + i32::from(delta_q_dc), 0, 255) as usize;
        match hdr.bit_depth {
            BitDepth::Depth8 => Ok(i32::from(DC_QLOOKUP[q_table_idx])),
            BitDepth::Depth10 => Ok(i32::from(DC_QLOOKUP_10[q_table_idx])),
            BitDepth::Depth12 => Ok(i32::from(DC_QLOOKUP_12[q_table_idx])),
        }
    }

    /// An implementation of get_ac_quant as per "8.6.1 Dequantization functions"
    fn get_ac_quant(hdr: &Header, segment_id: u8, luma: bool) -> Result<i32> {
        let delta_q_ac = if luma { 0 } else { hdr.quant.delta_q_uv_ac };
        let qindex = Self::get_qindex(hdr, segment_id);
        let q_table_idx = usize::try_from(Self::clamp(qindex + i32::from(delta_q_ac), 0, 255))?;

        match hdr.bit_depth {
            BitDepth::Depth8 => Ok(i32::from(AC_QLOOKUP[q_table_idx])),
            BitDepth::Depth10 => Ok(i32::from(AC_QLOOKUP_10[q_table_idx])),
            BitDepth::Depth12 => Ok(i32::from(AC_QLOOKUP_12[q_table_idx])),
        }
    }

    /// Update the state of the segmentation parameters after seeing a frame
    pub(crate) fn update_segmentation(
        hdr: &Header,
        segmentation: &mut [Segmentation; MAX_SEGMENTS],
    ) -> Result<()> {
        let lf = &hdr.lf;
        let seg = &hdr.seg;

        let n_shift = lf.level >> 5;

        for segment_id in 0..MAX_SEGMENTS as u8 {
            let luma_dc_quant_scale = i16::try_from(Self::get_dc_quant(hdr, segment_id, true)?)?;
            let luma_ac_quant_scale = i16::try_from(Self::get_ac_quant(hdr, segment_id, true)?)?;
            let chroma_dc_quant_scale = i16::try_from(Self::get_dc_quant(hdr, segment_id, false)?)?;
            let chroma_ac_quant_scale = i16::try_from(Self::get_ac_quant(hdr, segment_id, false)?)?;

            let mut lvl_seg = i32::from(lf.level);
            let mut lvl_lookup: [[u8; MAX_MODE_LF_DELTAS]; MAX_REF_FRAMES];

            if lvl_seg == 0 {
                lvl_lookup = Default::default()
            } else {
                // 8.8.1 Loop filter frame init process
                if Self::seg_feature_active(hdr, segment_id, SEG_LVL_ALT_L as u8) {
                    if seg.abs_or_delta_update {
                        lvl_seg =
                            i32::from(seg.feature_data[usize::from(segment_id)][SEG_LVL_ALT_L]);
                    } else {
                        lvl_seg +=
                            i32::from(seg.feature_data[usize::from(segment_id)][SEG_LVL_ALT_L]);
                    }

                    lvl_seg = Self::clamp(lvl_seg, 0, MAX_LOOP_FILTER as i32);
                }

                if !lf.delta_enabled {
                    lvl_lookup = [[u8::try_from(lvl_seg)?; MAX_MODE_LF_DELTAS]; MAX_REF_FRAMES]
                } else {
                    let intra_delta = i32::from(lf.ref_deltas[INTRA_FRAME]);
                    let mut intra_lvl = lvl_seg + (intra_delta << n_shift);

                    lvl_lookup = segmentation[usize::from(segment_id)].lvl_lookup;
                    lvl_lookup[INTRA_FRAME][0] =
                        u8::try_from(Self::clamp(intra_lvl, 0, MAX_LOOP_FILTER as i32))?;

                    // Note, this array has the [0] element unspecified/unused in
                    // VP9. Confusing, but we do start to index from 1.
                    #[allow(clippy::needless_range_loop)]
                    for ref_ in LAST_FRAME..MAX_REF_FRAMES {
                        for mode in 0..MAX_MODE_LF_DELTAS {
                            let ref_delta = i32::from(lf.ref_deltas[ref_]);
                            let mode_delta = i32::from(lf.mode_deltas[mode]);

                            intra_lvl = lvl_seg + (ref_delta << n_shift) + (mode_delta << n_shift);

                            lvl_lookup[ref_][mode] =
                                u8::try_from(Self::clamp(intra_lvl, 0, MAX_LOOP_FILTER as i32))?;
                        }
                    }
                }
            }

            segmentation[usize::from(segment_id)] = Segmentation {
                lvl_lookup,
                luma_ac_quant_scale,
                luma_dc_quant_scale,
                chroma_ac_quant_scale,
                chroma_dc_quant_scale,
                reference_frame_enabled: seg.feature_enabled[usize::from(segment_id)]
                    [SEG_LVL_REF_FRAME],
                reference_frame: seg.feature_data[usize::from(segment_id)][SEG_LVL_REF_FRAME],
                reference_skip_enabled: seg.feature_enabled[usize::from(segment_id)][SEG_LVL_SKIP],
            }
        }

        Ok(())
    }

    #[cfg(test)]
    #[allow(dead_code)]
    pub(crate) fn backend(&self) -> &dyn StatelessDecoderBackend<Handle = T> {
        self.backend.as_ref()
    }
}

impl<T: DecodedHandle + Clone + 'static> VideoDecoder for Decoder<T> {
    fn decode(
        &mut self,
        timestamp: u64,
        bitstream: &[u8],
    ) -> VideoDecoderResult<Vec<Box<dyn DecodedHandle>>> {
        let frames = self
            .parser
            .parse_chunk(|| bitstream)
            .map_err(|e| anyhow!(e))?;

        for frame in frames {
            if matches!(frame.header.frame_type, FrameType::KeyFrame) {
                if self.negotiation_possible(&frame)
                    && matches!(self.negotiation_status, NegotiationStatus::Negotiated)
                {
                    self.negotiation_status = NegotiationStatus::NonNegotiated;
                }
            }

            match &mut self.negotiation_status {
                NegotiationStatus::NonNegotiated => {
                    if matches!(frame.header.frame_type, FrameType::KeyFrame) {
                        self.backend.poll(BlockingMode::Blocking)?;

                        self.backend.new_sequence(&frame.header)?;

                        let offset = frame.offset;
                        let size = frame.size;
                        let bitstream = frame.bitstream[offset..offset + size].to_vec();

                        self.coded_resolution = Resolution {
                            width: frame.header.width,
                            height: frame.header.height,
                        };

                        self.profile = frame.header.profile;
                        self.bit_depth = frame.header.bit_depth;

                        self.negotiation_status = NegotiationStatus::Possible {
                            key_frame: (timestamp, Box::new(frame.header), bitstream),
                        }
                    }

                    return Ok(vec![]);
                }

                NegotiationStatus::Possible { key_frame } => {
                    let (timestamp, header, bitstream) = key_frame.clone();
                    let key_frame = Frame::new(bitstream.as_ref(), *header, 0, bitstream.len());

                    self.handle_frame(&key_frame, timestamp)?;

                    self.negotiation_status = NegotiationStatus::Negotiated;
                }

                NegotiationStatus::Negotiated => (),
            }

            self.handle_frame(&frame, timestamp)?;

            if self.backend.num_resources_left() == 0 {
                self.block_on_one()?;
            }

            self.backend.poll(self.blocking_mode)?;
        }

        let ready_frames = self.get_ready_frames();

        Ok(ready_frames
            .into_iter()
            .map(|h| Box::new(h) as Box<dyn DecodedHandle>)
            .collect())
    }

    fn flush(&mut self) -> VideoDecoderResult<Vec<Box<dyn DecodedHandle>>> {
        // Decode whatever is pending using the default format. Mainly covers
        // the rare case where only one buffer is sent.
        if let NegotiationStatus::Possible { key_frame } = &self.negotiation_status {
            let (timestamp, header, bitstream) = key_frame;

            let bitstream = bitstream.clone();
            let sz = bitstream.len();

            let header = header.as_ref().clone();

            let key_frame = Frame::new(bitstream.as_ref(), header, 0, sz);
            let timestamp = *timestamp;

            self.handle_frame(&key_frame, timestamp)?;
        }

        self.backend.poll(BlockingMode::Blocking)?;

        let pics = self.get_ready_frames();

        Ok(pics
            .into_iter()
            .map(|h| Box::new(h) as Box<dyn DecodedHandle>)
            .collect())
    }

    fn negotiation_possible(&self) -> bool {
        matches!(self.negotiation_status, NegotiationStatus::Possible { .. })
    }

    fn num_resources_left(&self) -> Option<usize> {
        if matches!(self.negotiation_status, NegotiationStatus::NonNegotiated) {
            return None;
        }

        let left_in_the_backend = self.backend.num_resources_left();

        if let NegotiationStatus::Possible { .. } = &self.negotiation_status {
            Some(left_in_the_backend - 1)
        } else {
            Some(left_in_the_backend)
        }
    }

    fn num_resources_total(&self) -> usize {
        self.backend.num_resources_total()
    }

    fn coded_resolution(&self) -> Option<Resolution> {
        self.backend.coded_resolution()
    }

    fn poll(
        &mut self,
        blocking_mode: BlockingMode,
    ) -> VideoDecoderResult<Vec<Box<dyn DecodedHandle>>> {
        let handles = self.backend.poll(blocking_mode)?;

        Ok(handles
            .into_iter()
            .map(|h| Box::new(h) as Box<dyn DecodedHandle>)
            .collect())
    }
}

#[cfg(test)]
pub mod tests {
    use std::io::Cursor;
    use std::io::Read;
    use std::io::Seek;

    use bytes::Buf;

    use crate::decoders::tests::test_decode_stream;
    use crate::decoders::tests::TestStream;
    use crate::decoders::vp9::decoder::Decoder;
    use crate::decoders::BlockingMode;
    use crate::decoders::DecodedHandle;
    use crate::decoders::VideoDecoder;

    /// Read and return the data from the next IVF packet. Returns `None` if there is no more data
    /// to read.
    pub(crate) fn read_ivf_packet(cursor: &mut Cursor<&[u8]>) -> Option<Box<[u8]>> {
        if !cursor.has_remaining() {
            return None;
        }

        let len = cursor.get_u32_le();
        // Skip PTS.
        let _ = cursor.get_u64_le();

        let mut buf = vec![0u8; len as usize];
        cursor.read_exact(&mut buf).unwrap();

        Some(buf.into_boxed_slice())
    }

    pub fn vp9_decoding_loop<D>(
        decoder: &mut D,
        test_stream: &[u8],
        on_new_frame: &mut dyn FnMut(Box<dyn DecodedHandle>),
    ) where
        D: VideoDecoder,
    {
        let mut cursor = Cursor::new(test_stream);
        let mut frame_num = 0;

        // Skip the IVH header entirely.
        cursor.seek(std::io::SeekFrom::Start(32)).unwrap();

        while let Some(packet) = read_ivf_packet(&mut cursor) {
            let bitstream = packet.as_ref();
            for frame in decoder.decode(frame_num, bitstream).unwrap() {
                on_new_frame(frame);
                frame_num += 1;
            }
        }

        for frame in decoder.flush().unwrap() {
            on_new_frame(frame);
            frame_num += 1;
        }
    }

    /// Run `test` using the dummy decoder, in both blocking and non-blocking modes.
    fn test_decoder_dummy(test: &TestStream, blocking_mode: BlockingMode) {
        let decoder = Decoder::new_dummy(blocking_mode).unwrap();

        test_decode_stream(vp9_decoding_loop, decoder, test, false, false);
    }

    /// Same as Chromium's test-25fps.vp8
    pub const DECODE_TEST_25FPS: TestStream = TestStream {
        stream: include_bytes!("test_data/test-25fps.vp9"),
        crcs: include_str!("test_data/test-25fps.vp9.crc"),
    };

    #[test]
    fn test_25fps_block() {
        test_decoder_dummy(&DECODE_TEST_25FPS, BlockingMode::Blocking);
    }

    #[test]
    fn test_25fps_nonblock() {
        test_decoder_dummy(&DECODE_TEST_25FPS, BlockingMode::NonBlocking);
    }

    // Remuxed from the original matroska source in libvpx using ffmpeg:
    // ffmpeg -i vp90-2-10-show-existing-frame.webm/vp90-2-10-show-existing-frame.webm -c:v copy /tmp/vp90-2-10-show-existing-frame.vp9.ivf
    pub const DECODE_TEST_25FPS_SHOW_EXISTING_FRAME: TestStream = TestStream {
        stream: include_bytes!("test_data/vp90-2-10-show-existing-frame.vp9.ivf"),
        crcs: include_str!("test_data/vp90-2-10-show-existing-frame.vp9.ivf.crc"),
    };

    #[test]
    fn show_existing_frame_block() {
        test_decoder_dummy(
            &DECODE_TEST_25FPS_SHOW_EXISTING_FRAME,
            BlockingMode::Blocking,
        );
    }

    #[test]
    fn show_existing_frame_nonblock() {
        test_decoder_dummy(
            &DECODE_TEST_25FPS_SHOW_EXISTING_FRAME,
            BlockingMode::NonBlocking,
        );
    }

    pub const DECODE_TEST_25FPS_SHOW_EXISTING_FRAME2: TestStream = TestStream {
        stream: include_bytes!("test_data/vp90-2-10-show-existing-frame2.vp9.ivf"),
        crcs: include_str!("test_data/vp90-2-10-show-existing-frame2.vp9.ivf.crc"),
    };

    #[test]
    fn show_existing_frame2_block() {
        test_decoder_dummy(
            &DECODE_TEST_25FPS_SHOW_EXISTING_FRAME2,
            BlockingMode::Blocking,
        );
    }

    #[test]
    fn show_existing_frame2_nonblock() {
        test_decoder_dummy(
            &DECODE_TEST_25FPS_SHOW_EXISTING_FRAME2,
            BlockingMode::NonBlocking,
        );
    }

    // Remuxed from the original matroska source in libvpx using ffmpeg:
    // ffmpeg -i vp90-2-10-show-existing-frame.webm/vp90-2-10-show-existing-frame.webm -c:v copy /tmp/vp90-2-10-show-existing-frame.vp9.ivf
    // There are some weird padding issues introduced by GStreamer for
    // resolutions that are not multiple of 4, so we're ignoring CRCs for
    // this one.
    pub const DECODE_RESOLUTION_CHANGE_500FRAMES: TestStream = TestStream {
        stream: include_bytes!("test_data/resolution_change_500frames-vp9.ivf"),
        crcs: include_str!("test_data/resolution_change_500frames-vp9.ivf.crc"),
    };

    #[test]
    fn test_resolution_change_500frames_block() {
        test_decoder_dummy(&DECODE_RESOLUTION_CHANGE_500FRAMES, BlockingMode::Blocking);
    }

    #[test]
    fn test_resolution_change_500frames_nonblock() {
        test_decoder_dummy(&DECODE_RESOLUTION_CHANGE_500FRAMES, BlockingMode::Blocking);
    }
}
