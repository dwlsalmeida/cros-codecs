// Copyright 2022 The ChromiumOS Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

use anyhow::anyhow;

use crate::decoders::vp8::backends::StatelessDecoderBackend;
use crate::decoders::vp8::parser::Frame;
use crate::decoders::vp8::parser::Header;
use crate::decoders::vp8::parser::Parser;
use crate::decoders::BlockingMode;
use crate::decoders::DecodedHandle;
use crate::decoders::ReadyFramesQueue;
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
        key_frame: (u64, Box<Header>, Vec<u8>, Box<Parser>),
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

    ready_queue: ReadyFramesQueue<T>,

    /// The picture used as the last reference picture.
    last_picture: Option<T>,
    /// The picture used as the golden reference picture.
    golden_ref_picture: Option<T>,
    /// The picture used as the alternate reference picture.
    alt_ref_picture: Option<T>,
}

impl<T: DecodedHandle + Clone + 'static> Decoder<T> {
    /// Create a new codec backend for VP8.
    #[cfg(any(feature = "vaapi", test))]
    pub(crate) fn new(
        backend: Box<dyn StatelessDecoderBackend<Handle = T>>,
        blocking_mode: BlockingMode,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            backend,
            blocking_mode,
            // wait_keyframe: true,
            parser: Default::default(),
            negotiation_status: Default::default(),
            last_picture: Default::default(),
            golden_ref_picture: Default::default(),
            alt_ref_picture: Default::default(),
            coded_resolution: Default::default(),
            ready_queue: Default::default(),
        })
    }

    /// Replace a reference frame with `handle`.
    fn replace_reference(reference: &mut Option<T>, handle: &T) {
        *reference = Some(handle.clone());
    }

    pub(crate) fn update_references(
        header: &Header,
        decoded_handle: &T,
        last_picture: &mut Option<T>,
        golden_ref_picture: &mut Option<T>,
        alt_ref_picture: &mut Option<T>,
    ) -> anyhow::Result<()> {
        if header.key_frame() {
            Decoder::replace_reference(last_picture, decoded_handle);
            Decoder::replace_reference(golden_ref_picture, decoded_handle);
            Decoder::replace_reference(alt_ref_picture, decoded_handle);
        } else {
            if header.refresh_alternate_frame() {
                Decoder::replace_reference(alt_ref_picture, decoded_handle);
            } else {
                match header.copy_buffer_to_alternate() {
                    0 => { /* do nothing */ }

                    1 => {
                        if let Some(last_picture) = last_picture {
                            Decoder::replace_reference(alt_ref_picture, last_picture);
                        }
                    }

                    2 => {
                        if let Some(golden_ref) = golden_ref_picture {
                            Decoder::replace_reference(alt_ref_picture, golden_ref);
                        }
                    }

                    other => panic!("Invalid value: {}", other),
                }
            }

            if header.refresh_golden_frame() {
                Decoder::replace_reference(golden_ref_picture, decoded_handle);
            } else {
                match header.copy_buffer_to_golden() {
                    0 => { /* do nothing */ }

                    1 => {
                        if let Some(last_picture) = last_picture {
                            Decoder::replace_reference(golden_ref_picture, last_picture);
                        }
                    }

                    2 => {
                        if let Some(alt_ref) = alt_ref_picture {
                            Decoder::replace_reference(golden_ref_picture, alt_ref);
                        }
                    }

                    other => panic!("Invalid value: {}", other),
                }
            }

            if header.refresh_last() {
                Decoder::replace_reference(last_picture, decoded_handle);
            }
        }

        Ok(())
    }

    fn block_on_one(&mut self) -> anyhow::Result<()> {
        if let Some(handle) = self.ready_queue.peek() {
            return handle.sync().map_err(|e| e.into());
        }

        Ok(())
    }

    /// Handle a single frame.
    fn handle_frame(
        &mut self,
        frame: Frame<&[u8]>,
        timestamp: u64,
        queued_parser_state: Option<Parser>,
    ) -> anyhow::Result<()> {
        let parser = match &queued_parser_state {
            Some(parser) => parser,
            None => &self.parser,
        };

        let block = matches!(self.blocking_mode, BlockingMode::Blocking)
            || matches!(self.negotiation_status, NegotiationStatus::Possible { .. });

        let show_frame = frame.header.show_frame();

        let decoded_handle = self
            .backend
            .submit_picture(
                &frame.header,
                self.last_picture.as_ref(),
                self.golden_ref_picture.as_ref(),
                self.alt_ref_picture.as_ref(),
                frame.bitstream,
                parser.segmentation(),
                parser.mb_lf_adjust(),
                timestamp,
            )
            .map_err(|e| anyhow!(e))?;

        if block {
            decoded_handle.sync()?;
        }

        // Do DPB management
        Self::update_references(
            &frame.header,
            &decoded_handle,
            &mut self.last_picture,
            &mut self.golden_ref_picture,
            &mut self.alt_ref_picture,
        )?;

        if show_frame {
            self.ready_queue.push(decoded_handle);
        }

        Ok(())
    }

    fn negotiation_possible(&self, frame: &Frame<impl AsRef<[u8]>>) -> bool {
        let coded_resolution = self.coded_resolution;
        let hdr = &frame.header;
        let width = u32::from(hdr.width());
        let height = u32::from(hdr.height());

        width != coded_resolution.width || height != coded_resolution.height
    }
}

impl<T: DecodedHandle + Clone + 'static> VideoDecoder for Decoder<T> {
    fn decode(
        &mut self,
        timestamp: u64,
        bitstream: &[u8],
    ) -> VideoDecoderResult<Vec<Box<dyn DecodedHandle>>> {
        let frame = self.parser.parse_frame(bitstream).map_err(|e| anyhow!(e))?;

        if frame.header.key_frame()
            && self.negotiation_possible(&frame)
            && matches!(self.negotiation_status, NegotiationStatus::Negotiated)
        {
            self.negotiation_status = NegotiationStatus::NonNegotiated;
        }

        match &mut self.negotiation_status {
            NegotiationStatus::NonNegotiated => {
                if frame.header.key_frame() {
                    self.backend.new_sequence(&frame.header)?;

                    self.coded_resolution = Resolution {
                        width: u32::from(frame.header.width()),
                        height: u32::from(frame.header.height()),
                    };

                    self.negotiation_status = NegotiationStatus::Possible {
                        key_frame: (
                            timestamp,
                            Box::new(frame.header),
                            Vec::from(frame.bitstream),
                            Box::new(self.parser.clone()),
                        ),
                    }
                }

                return Ok(vec![]);
            }

            NegotiationStatus::Possible { key_frame } => {
                let (timestamp, header, bitstream, parser) = key_frame.clone();
                let key_frame = Frame {
                    bitstream: bitstream.as_ref(),
                    header: *header,
                };

                self.handle_frame(key_frame, timestamp, Some(*parser))?;

                self.negotiation_status = NegotiationStatus::Negotiated;
            }

            NegotiationStatus::Negotiated => (),
        };

        self.handle_frame(frame, timestamp, None)?;

        if self.backend.num_resources_left() == 0 {
            self.block_on_one()?;
        }

        Ok(self
            .ready_queue
            .get_ready_frames()?
            .into_iter()
            .map(|h| Box::new(h) as Box<dyn DecodedHandle>)
            .collect())
    }

    fn flush(&mut self) -> crate::decoders::Result<Vec<Box<dyn DecodedHandle>>> {
        // Decode whatever is pending using the default format. Mainly covers
        // the rare case where only one buffer is sent.
        if let NegotiationStatus::Possible { key_frame } = &self.negotiation_status {
            let (timestamp, header, bitstream, parser) = key_frame;

            let bitstream = bitstream.clone();
            let header = header.as_ref().clone();

            let key_frame = Frame {
                bitstream: bitstream.as_ref(),
                header,
            };
            let timestamp = *timestamp;
            let parser = *parser.clone();

            self.handle_frame(key_frame, timestamp, Some(parser))?;
        }

        // Make sure all frames will be output.
        for handle in &mut self.ready_queue {
            handle.sync()?;
        }

        Ok(self
            .ready_queue
            .get_ready_frames()?
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
}

#[cfg(test)]
pub mod tests {
    use std::io::Cursor;
    use std::io::Seek;

    use crate::decoders::tests::test_decode_stream;
    use crate::decoders::tests::TestStream;
    use crate::decoders::vp8::decoder::Decoder;
    use crate::decoders::BlockingMode;
    use crate::decoders::DecodedHandle;
    use crate::decoders::VideoDecoder;
    use crate::utils::read_ivf_packet;

    pub fn vp8_decoding_loop<D>(
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
            for frame in decoder.decode(frame_num, packet.as_ref()).unwrap() {
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

        test_decode_stream(vp8_decoding_loop, decoder, test, false, false);
    }

    /// Same as Chromium's test-25fps.vp8
    pub const DECODE_TEST_25FPS: TestStream = TestStream {
        stream: include_bytes!("test_data/test-25fps.vp8"),
        crcs: include_str!("test_data/test-25fps.vp8.crc"),
    };

    #[test]
    fn test_25fps_block() {
        test_decoder_dummy(&DECODE_TEST_25FPS, BlockingMode::Blocking);
    }

    #[test]
    fn test_25fps_nonblock() {
        test_decoder_dummy(&DECODE_TEST_25FPS, BlockingMode::NonBlocking);
    }
}
