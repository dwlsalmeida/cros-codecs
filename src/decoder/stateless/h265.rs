// Copyright 2023 The ChromiumOS Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#[cfg(test)]
mod dummy;
#[cfg(feature = "vaapi")]
mod vaapi;

use std::cell::RefCell;
use std::io::Cursor;
use std::rc::Rc;

use anyhow::anyhow;
use anyhow::Context;

use crate::codec::h265::dpb::Dpb;
use crate::codec::h265::dpb::DpbEntry;
use crate::codec::h265::parser::Nalu;
use crate::codec::h265::parser::NaluType;
use crate::codec::h265::parser::Parser;
use crate::codec::h265::parser::Pps;
use crate::codec::h265::parser::ShortTermRefPicSet;
use crate::codec::h265::parser::Slice;
use crate::codec::h265::parser::SliceHeader;
use crate::codec::h265::parser::Sps;
use crate::codec::h265::picture::PictureData;
use crate::codec::h265::picture::Reference;
use crate::decoder::stateless::private;
use crate::decoder::stateless::DecodeError;
use crate::decoder::stateless::DecodingState;
use crate::decoder::stateless::StatelessBackendResult;
use crate::decoder::stateless::StatelessDecoderBackend;
use crate::decoder::stateless::StatelessDecoderFormatNegotiator;
use crate::decoder::stateless::StatelessVideoDecoder;
use crate::decoder::BlockingMode;
use crate::decoder::DecodedHandle;
use crate::decoder::DecoderEvent;
use crate::decoder::ReadyFramesQueue;
use crate::decoder::StreamInfo;
use crate::decoder::SurfacePool;
use crate::Resolution;

const MAX_DPB_SIZE: usize = 16;

// Equation 5-8
pub(crate) fn clip3(x: i32, y: i32, z: i32) -> i32 {
    if z < x {
        x
    } else if z > y {
        y
    } else {
        z
    }
}

// See 6.5.3
const fn up_right_diagonal<const N: usize, const ROWS: usize>() -> [usize; N] {
    // Generics can't be used in const operations for now, so [0; ROWS * ROWS]
    // is rejected by the compiler
    assert!(ROWS * ROWS == N);

    let mut i = 0;
    let mut x = 0i32;
    let mut y = 0i32;
    let mut ret = [0; N];

    loop {
        while y >= 0 {
            if x < (ROWS as i32) && y < (ROWS as i32) {
                ret[i] = (x + ROWS as i32 * y) as usize;
                i += 1;
            }
            y -= 1;
            x += 1;
        }

        y = x;
        x = 0;
        if i >= N {
            break;
        }
    }

    ret
}

const UP_RIGHT_DIAGONAL_4X4: [usize; 16] = up_right_diagonal::<16, 4>();
const UP_RIGHT_DIAGONAL_8X8: [usize; 64] = up_right_diagonal::<64, 8>();

fn get_raster_from_up_right_diagonal_8x8(src: [u8; 64], dst: &mut [u8; 64]) {
    for i in 0..64 {
        dst[UP_RIGHT_DIAGONAL_8X8[i]] = src[i];
    }
}

fn get_raster_from_up_right_diagonal_4x4(src: [u8; 16], dst: &mut [u8; 16]) {
    for i in 0..16 {
        dst[UP_RIGHT_DIAGONAL_4X4[i]] = src[i];
    }
}

/// Stateless backend methods specific to H.265.
trait StatelessH265DecoderBackend: StatelessDecoderBackend<Sps> {
    /// Called when a new SPS is parsed.
    fn new_sequence(&mut self, sps: &Sps) -> StatelessBackendResult<()>;

    /// Called when the decoder determines that a frame or field was found.
    fn new_picture(
        &mut self,
        picture: &PictureData,
        timestamp: u64,
    ) -> StatelessBackendResult<Self::Picture>;

    /// Called by the decoder for every frame or field found.
    #[allow(clippy::too_many_arguments)]
    fn handle_picture(
        &mut self,
        picture: &mut Self::Picture,
        picture_data: &PictureData,
        sps: &Sps,
        pps: &Pps,
        dpb: &Dpb<Self::Handle>,
        rps: &RefPicSet<Self::Handle>,
        slice: &Slice<&[u8]>,
    ) -> StatelessBackendResult<()>;

    /// Called to dispatch a decode operation to the backend.
    #[allow(clippy::too_many_arguments)]
    fn decode_slice(
        &mut self,
        picture: &mut Self::Picture,
        slice: &Slice<&[u8]>,
        sps: &Sps,
        pps: &Pps,
        dpb: &Dpb<Self::Handle>,
        ref_pic_list0: &[Option<RefPicListEntry<Self::Handle>>; 16],
        ref_pic_list1: &[Option<RefPicListEntry<Self::Handle>>; 16],
    ) -> StatelessBackendResult<()>;

    /// Called when the decoder wants the backend to finish the decoding
    /// operations for `picture`. At this point, `decode_slice` has been called
    /// for all slices.
    ///
    /// This call will assign the ownership of the BackendHandle to the Picture
    /// and then assign the ownership of the Picture to the Handle.
    fn submit_picture(&mut self, picture: Self::Picture) -> StatelessBackendResult<Self::Handle>;
}

/// An entry in the Reference Picture Lists. Unlike H.264, H.265 can use the
/// current picture itself as a reference.
#[derive(Clone)]
pub enum RefPicListEntry<T> {
    CurrentPicture(PictureData),
    DpbEntry(DpbEntry<T>),
}

enum BumpingType {
    BeforeDecoding,
    AfterDecoding,
}

/// Keeps track of the last values seen for negotiation purposes.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct NegotiationInfo {
    /// The current coded resolution
    coded_resolution: Resolution,
    general_profile_idc: u8,
    bit_depth_luma_minus8: u8,
    bit_depth_chroma_minus8: u8,
    chroma_format_idc: u8,
}

impl From<&Sps> for NegotiationInfo {
    fn from(sps: &Sps) -> Self {
        NegotiationInfo {
            coded_resolution: Resolution {
                width: sps.width().into(),
                height: sps.height().into(),
            },
            general_profile_idc: sps.profile_tier_level().general_profile_idc(),
            bit_depth_luma_minus8: sps.bit_depth_luma_minus8(),
            bit_depth_chroma_minus8: sps.bit_depth_chroma_minus8(),
            chroma_format_idc: sps.chroma_format_idc(),
        }
    }
}

/// The RefPicSet data, derived once per picture.
#[derive(Clone, Debug)]
struct RefPicSet<T> {
    curr_delta_poc_msb_present_flag: [bool; MAX_DPB_SIZE],
    foll_delta_poc_msb_present_flag: [bool; MAX_DPB_SIZE],

    num_poc_st_curr_before: usize,
    num_poc_st_curr_after: usize,
    num_poc_st_foll: usize,
    num_poc_lt_curr: usize,
    num_poc_lt_foll: usize,

    poc_st_curr_before: [i32; MAX_DPB_SIZE],
    poc_st_curr_after: [i32; MAX_DPB_SIZE],
    poc_st_foll: [i32; MAX_DPB_SIZE],
    poc_lt_curr: [i32; MAX_DPB_SIZE],
    poc_lt_foll: [i32; MAX_DPB_SIZE],

    ref_pic_set_lt_curr: [Option<DpbEntry<T>>; MAX_DPB_SIZE],
    ref_pic_set_st_curr_after: [Option<DpbEntry<T>>; MAX_DPB_SIZE],
    ref_pic_set_st_curr_before: [Option<DpbEntry<T>>; MAX_DPB_SIZE],
    ref_pic_set_st_foll: [Option<DpbEntry<T>>; MAX_DPB_SIZE],
    ref_pic_set_lt_foll: [Option<DpbEntry<T>>; MAX_DPB_SIZE],
}

impl<T: Clone> Default for RefPicSet<T> {
    fn default() -> Self {
        Self {
            curr_delta_poc_msb_present_flag: Default::default(),
            foll_delta_poc_msb_present_flag: Default::default(),
            num_poc_st_curr_before: Default::default(),
            num_poc_st_curr_after: Default::default(),
            num_poc_st_foll: Default::default(),
            num_poc_lt_curr: Default::default(),
            num_poc_lt_foll: Default::default(),
            poc_st_curr_before: Default::default(),
            poc_st_curr_after: Default::default(),
            poc_st_foll: Default::default(),
            poc_lt_curr: Default::default(),
            poc_lt_foll: Default::default(),
            ref_pic_set_lt_curr: Default::default(),
            ref_pic_set_st_curr_after: Default::default(),
            ref_pic_set_st_curr_before: Default::default(),
            ref_pic_set_st_foll: Default::default(),
            ref_pic_set_lt_foll: Default::default(),
        }
    }
}

pub struct Decoder<T, P>
where
    T: Clone,
{
    /// A parser to extract bitstream metadata
    parser: Parser,

    /// Whether the decoder should block on decode operations.
    blocking_mode: BlockingMode,

    /// The backend used for hardware acceleration.
    backend: Box<dyn StatelessH265DecoderBackend<Handle = T, Picture = P>>,

    /// The backend used for hardware acceleration.
    // backend: Box<dyn StatelessDecoderBackend<Handle = T, Picture = P>>,
    decoding_state: DecodingState<Sps>,

    /// Keeps track of the last values seen for negotiation purposes.
    negotiation_info: NegotiationInfo,

    rps: RefPicSet<T>,

    ready_queue: ReadyFramesQueue<T>,

    /// The decoded picture buffer
    dpb: Dpb<T>,

    /// The current active SPS id.
    cur_sps_id: u8,
    /// The current active PPS id.
    cur_pps_id: u8,

    /// The current picture being worked on. It is set during `handle_picture()`
    /// and passed to the accelerator during `submit_picture()`.
    cur_pic: Option<PictureData>,
    /// The current picture representation on the backend side.
    cur_backend_pic: Option<P>,

    /// Used to identify first picture in decoding order or first picture that
    /// follows an EOS NALU.
    first_picture_after_eos: bool,

    /// Whether this is the first picture in the bitstream in decoding order.
    first_picture_in_bitstream: bool,
    // Same as PrevTid0Pic in the specification.
    prev_tid_0_pic: Option<PictureData>,

    // Internal variables needed during the decoding process.
    max_pic_order_cnt_lsb: i32,
    /// The value of NoRaslOutputFlag for the last IRAP picture.
    irap_no_rasl_output_flag: bool,

    /// Reference picture list 0 for P and B slices. Retains the same meaning as
    /// in the specification. Points into the pictures stored in the DPB.
    /// Derived once per slice.
    ref_pic_list0: [Option<RefPicListEntry<T>>; MAX_DPB_SIZE],
    /// Reference picture list 1 for B slices. Retains the same meaning as in
    /// the specification. Points into the pictures stored in the DPB. Derived
    /// once per slice.
    ref_pic_list1: [Option<RefPicListEntry<T>>; MAX_DPB_SIZE],
    /// We keep track of the last independent header so we can copy that into
    /// dependent slices.
    last_independent_slice_header: Option<SliceHeader>,
}

impl<T, P> Decoder<T, P>
where
    T: DecodedHandle + Clone,
{
    /// Create a new decoder using the given `backend`.
    #[cfg(any(feature = "vaapi", test))]
    #[allow(dead_code)]
    fn new(
        backend: Box<dyn StatelessH265DecoderBackend<Handle = T, Picture = P>>,
        blocking_mode: BlockingMode,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            backend,
            blocking_mode,
            first_picture_after_eos: true,
            first_picture_in_bitstream: true,
            parser: Default::default(),
            decoding_state: Default::default(),
            negotiation_info: Default::default(),
            rps: Default::default(),
            ready_queue: Default::default(),
            cur_sps_id: Default::default(),
            cur_pps_id: Default::default(),
            cur_pic: Default::default(),
            prev_tid_0_pic: Default::default(),
            max_pic_order_cnt_lsb: Default::default(),
            irap_no_rasl_output_flag: Default::default(),
            ref_pic_list0: Default::default(),
            ref_pic_list1: Default::default(),
            cur_backend_pic: Default::default(),
            dpb: Default::default(),
            last_independent_slice_header: Default::default(),
        })
    }

    /// Whether the stream parameters have changed, indicating that a negotiation window has opened.
    fn negotiation_possible(
        sps: &Sps,
        dpb: &Dpb<T>,
        old_negotiation_info: &NegotiationInfo,
    ) -> bool {
        let negotiation_info = NegotiationInfo::from(sps);
        let max_dpb_size = std::cmp::min(sps.max_dpb_size(), 16);
        let prev_max_dpb_size = dpb.max_num_pics();

        *old_negotiation_info != negotiation_info || prev_max_dpb_size != max_dpb_size
    }

    fn peek_sps(bitstream: &[u8]) -> anyhow::Result<Option<Sps>> {
        let mut cursor = Cursor::new(bitstream);

        while let Ok(nalu) = Nalu::next(&mut cursor) {
            if matches!(nalu.header().type_(), NaluType::SpsNut) {
                let sps = Parser::peek_sps(&nalu)?;
                return Ok(Some(sps));
            }
        }

        Ok(None)
    }

    /// Apply the parameters of `sps` to the decoder.
    fn apply_sps(&mut self, sps: &Sps) {
        self.drain();
        self.negotiation_info = NegotiationInfo::from(sps);

        let max_dpb_size = std::cmp::min(sps.max_dpb_size(), 16);
        self.dpb.set_max_num_pics(max_dpb_size);
    }

    // See 8.3.2, Note 2.
    fn st_ref_pic_set<'a>(hdr: &'a SliceHeader, sps: &'a Sps) -> &'a ShortTermRefPicSet {
        if hdr.curr_rps_idx() == sps.num_short_term_ref_pic_sets() {
            hdr.short_term_ref_pic_set()
        } else {
            &sps.short_term_ref_pic_set()[usize::from(hdr.curr_rps_idx())]
        }
    }

    // See 8.3.2.
    fn decode_rps(&mut self, slice: &Slice<&[u8]>) -> anyhow::Result<()> {
        let hdr = slice.header();
        let cur_pic = self.cur_pic.as_ref().unwrap();

        if cur_pic.nalu_type.is_irap() && cur_pic.no_rasl_output_flag {
            self.dpb.mark_all_as_unused_for_ref();
        }

        if slice.nalu().header().type_().is_idr() {
            self.rps.poc_st_curr_before = Default::default();
            self.rps.poc_st_curr_after = Default::default();
            self.rps.poc_st_foll = Default::default();
            self.rps.poc_lt_curr = Default::default();
            self.rps.poc_lt_foll = Default::default();

            self.rps.num_poc_st_curr_before = 0;
            self.rps.num_poc_st_curr_after = 0;
            self.rps.num_poc_st_foll = 0;
            self.rps.num_poc_lt_curr = 0;
            self.rps.num_poc_lt_foll = 0;
        } else {
            // Equation 8-5
            let pps = self
                .parser
                .get_pps(hdr.pic_parameter_set_id())
                .context("Invalid SPS in init_current_pic")?;

            let sps = self
                .parser
                .get_sps(pps.seq_parameter_set_id())
                .context("Invalid PPS in init_current_pic")?;

            let curr_st_rps = Self::st_ref_pic_set(hdr, sps);
            let mut j = 0;
            let mut k = 0;
            for i in 0..usize::from(curr_st_rps.num_negative_pics()) {
                let poc = cur_pic.pic_order_cnt_val + curr_st_rps.delta_poc_s0()[i];

                if curr_st_rps.used_by_curr_pic_s0()[i] {
                    self.rps.poc_st_curr_before[j] = poc;
                    j += 1;
                } else {
                    self.rps.poc_st_foll[k] = poc;
                    k += 1;
                }
            }

            self.rps.num_poc_st_curr_before = j as _;

            let mut j = 0;
            for i in 0..usize::from(curr_st_rps.num_positive_pics()) {
                let poc = cur_pic.pic_order_cnt_val + curr_st_rps.delta_poc_s1()[i];

                if curr_st_rps.used_by_curr_pic_s1()[i] {
                    self.rps.poc_st_curr_after[j] = poc;
                    j += 1;
                } else {
                    self.rps.poc_st_foll[k] = poc;
                    k += 1;
                }
            }

            self.rps.num_poc_st_curr_after = j as _;
            self.rps.num_poc_st_foll = k as _;

            let mut j = 0;
            let mut k = 0;
            for i in 0..usize::from(hdr.num_long_term_sps() + hdr.num_long_term_pics()) {
                let mut poc_lt = hdr.poc_lsb_lt()[i] as i32;
                if hdr.delta_poc_msb_present_flag()[i] {
                    poc_lt += cur_pic.pic_order_cnt_val;
                    let delta_poc =
                        hdr.delta_poc_msb_cycle_lt()[i] as i32 * self.max_pic_order_cnt_lsb;

                    poc_lt -= delta_poc;
                    poc_lt -= cur_pic.pic_order_cnt_val & (self.max_pic_order_cnt_lsb - 1);
                }

                if hdr.used_by_curr_pic_lt()[i] {
                    self.rps.poc_lt_curr[j] = poc_lt;
                    self.rps.curr_delta_poc_msb_present_flag[j] =
                        hdr.delta_poc_msb_present_flag()[i];
                    j += 1;
                } else {
                    self.rps.poc_lt_foll[k] = poc_lt;
                    self.rps.foll_delta_poc_msb_present_flag[k] =
                        hdr.delta_poc_msb_present_flag()[i];
                    k += 1;
                }
            }

            self.rps.num_poc_lt_curr = j as _;
            self.rps.num_poc_lt_foll = k as _;
        }

        self.derive_and_mark_rps(hdr)?;
        Ok(())
    }

    // See the derivation process in the second half of 8.3.2.
    fn derive_and_mark_rps(&mut self, hdr: &SliceHeader) -> anyhow::Result<()> {
        let pps = self
            .parser
            .get_pps(hdr.pic_parameter_set_id())
            .context("Invalid SPS in init_current_pic")?;

        let sps = self
            .parser
            .get_sps(pps.seq_parameter_set_id())
            .context("Invalid PPS in init_current_pic")?;

        let max_pic_order_cnt_lsb = 2i32.pow((sps.log2_max_pic_order_cnt_lsb_minus4() + 4).into());

        // Equation 8-6
        for i in 0..self.rps.num_poc_lt_curr {
            if !self.rps.curr_delta_poc_msb_present_flag[i] {
                let poc = self.rps.poc_lt_curr[i];
                let mask = max_pic_order_cnt_lsb - 1;
                let reference = self.dpb.find_ref_by_poc_masked(poc, mask);

                if reference.is_none() {
                    log::warn!("No reference found for poc {} and mask {}", poc, mask);
                }

                self.rps.ref_pic_set_lt_curr[i] = reference;
            } else {
                let poc = self.rps.poc_lt_curr[i];
                let reference = self.dpb.find_ref_by_poc(poc);

                if reference.is_none() {
                    log::warn!("No reference found for poc {}", poc);
                }

                self.rps.ref_pic_set_lt_curr[i] = reference;
            }
        }

        for i in 0..self.rps.num_poc_lt_foll {
            if !self.rps.foll_delta_poc_msb_present_flag[i] {
                let poc = self.rps.poc_lt_foll[i];
                let mask = max_pic_order_cnt_lsb - 1;
                let reference = self.dpb.find_ref_by_poc_masked(poc, mask);

                if reference.is_none() {
                    log::warn!("No reference found for poc {} and mask {}", poc, mask);
                }

                self.rps.ref_pic_set_lt_foll[i] = reference;
            } else {
                let poc = self.rps.poc_lt_foll[i];
                let reference = self.dpb.find_ref_by_poc(poc);

                if reference.is_none() {
                    log::warn!("No reference found for poc {}", poc);
                }

                self.rps.ref_pic_set_lt_foll[i] = reference;
            }
        }

        for pic in self.rps.ref_pic_set_lt_curr.iter().flatten() {
            pic.0.borrow_mut().set_reference(Reference::LongTerm);
        }

        for pic in self.rps.ref_pic_set_lt_foll.iter().flatten() {
            pic.0.borrow_mut().set_reference(Reference::LongTerm);
        }

        // Equation 8-7
        for i in 0..self.rps.num_poc_st_curr_before {
            let poc = self.rps.poc_st_curr_before[i];
            let reference = self.dpb.find_short_term_ref_by_poc(poc);

            if reference.is_none() {
                log::warn!("No reference found for poc {}", poc);
            }

            self.rps.ref_pic_set_st_curr_before[i] = reference;
        }

        for i in 0..self.rps.num_poc_st_curr_after {
            let poc = self.rps.poc_st_curr_after[i];
            let reference = self.dpb.find_short_term_ref_by_poc(poc);

            if reference.is_none() {
                log::warn!("No reference found for poc {}", poc);
            }

            self.rps.ref_pic_set_st_curr_after[i] = reference;
        }

        for i in 0..self.rps.num_poc_st_foll {
            let poc = self.rps.poc_st_foll[i];
            let reference = self.dpb.find_short_term_ref_by_poc(poc);

            if reference.is_none() {
                log::warn!("No reference found for poc {}", poc);
            }

            self.rps.ref_pic_set_st_foll[i] = reference;
        }

        // 4. All reference pictures in the DPB that are not included in
        // RefPicSetLtCurr, RefPicSetLtFoll, RefPicSetStCurrBefore,
        // RefPicSetStCurrAfter, or RefPicSetStFoll and have nuh_layer_id equal
        // to currPicLayerId are marked as "unused for reference"
        for dpb_pic in self.dpb.entries() {
            let find_predicate = |p: &Option<DpbEntry<T>>| match p {
                Some(p) => p.0.borrow().pic_order_cnt_val == dpb_pic.0.borrow().pic_order_cnt_val,
                None => false,
            };

            if !self.rps.ref_pic_set_lt_curr[0..self.rps.num_poc_lt_curr]
                .iter()
                .any(find_predicate)
                && !self.rps.ref_pic_set_lt_foll[0..self.rps.num_poc_lt_foll]
                    .iter()
                    .any(find_predicate)
                && !self.rps.ref_pic_set_st_curr_after[0..self.rps.num_poc_st_curr_after]
                    .iter()
                    .any(find_predicate)
                && !self.rps.ref_pic_set_st_curr_before[0..self.rps.num_poc_st_curr_before]
                    .iter()
                    .any(find_predicate)
                && !self.rps.ref_pic_set_st_foll[0..self.rps.num_poc_st_foll]
                    .iter()
                    .any(find_predicate)
            {
                dpb_pic.0.borrow_mut().set_reference(Reference::None);
            }
        }

        let total_rps_len = self.rps.ref_pic_set_lt_curr[0..self.rps.num_poc_lt_curr].len()
            + self.rps.ref_pic_set_lt_foll[0..self.rps.num_poc_lt_foll].len()
            + self.rps.ref_pic_set_st_curr_after[0..self.rps.num_poc_st_curr_after].len()
            + self.rps.ref_pic_set_st_curr_before[0..self.rps.num_poc_st_curr_before].len()
            + self.rps.ref_pic_set_st_foll[0..self.rps.num_poc_st_foll].len();

        if self.dpb.entries().len() != total_rps_len {
            log::warn!("A reference pic is in more than one RPS list. This is against the specification. See 8.3.2. NOTE 5")
        }

        // According to Chromium, unavailable reference pictures are handled by
        // the accelerators internally.
        Ok(())
    }

    // See 8.3.4.
    // Builds the reference picture list for `hdr` for P and B slices.
    fn build_ref_pic_lists(&mut self, hdr: &SliceHeader) -> anyhow::Result<()> {
        // I slices do not use inter prediction.
        if !hdr.type_().is_p() && !hdr.type_().is_b() {
            return Ok(());
        }

        let pps = self
            .parser
            .get_pps(hdr.pic_parameter_set_id())
            .context("Invalid PPS in build_ref_pic_lists")?;

        if self.rps.num_poc_st_curr_before == 0
            && self.rps.num_poc_st_curr_after == 0
            && self.rps.num_poc_lt_curr == 0
            && pps.scc_extension_flag()
            && !pps.scc_extension().curr_pic_ref_enabled_flag()
        {
            // Let's try and keep going, if it is a broken stream then maybe it
            // will sort itself out as we go. In any case, we must not loop
            // infinitely here.
            log::error!("Bug or broken stream: out of pictures and can't build ref pic lists.");
            return Ok(());
        }

        let cur_pic = self.cur_pic.as_ref().unwrap();
        let rplm = hdr.ref_pic_list_modification();

        let num_rps_curr_temp_list0 = std::cmp::max(
            u32::from(hdr.num_ref_idx_l0_active_minus1()) + 1,
            hdr.num_pic_total_curr(),
        );

        // This could be simplified using a Vec, but lets not change the
        // algorithm from the spec too much.
        let mut ref_pic_list_temp0: [Option<RefPicListEntry<T>>; MAX_DPB_SIZE] = Default::default();

        // Equation 8-8
        let mut r_idx = 0;
        assert!(num_rps_curr_temp_list0 as usize <= MAX_DPB_SIZE);
        while r_idx < num_rps_curr_temp_list0 {
            let mut i = 0;
            while i < self.rps.num_poc_st_curr_before && r_idx < num_rps_curr_temp_list0 {
                ref_pic_list_temp0[r_idx as usize] = self.rps.ref_pic_set_st_curr_before[i]
                    .clone()
                    .map(|e| RefPicListEntry::DpbEntry(e));

                i += 1;
                r_idx += 1;
            }

            let mut i = 0;
            while i < self.rps.num_poc_st_curr_after && r_idx < num_rps_curr_temp_list0 {
                ref_pic_list_temp0[r_idx as usize] = self.rps.ref_pic_set_st_curr_after[i]
                    .clone()
                    .map(|e| RefPicListEntry::DpbEntry(e));

                i += 1;
                r_idx += 1;
            }

            let mut i = 0;
            while i < self.rps.num_poc_lt_curr && r_idx < num_rps_curr_temp_list0 {
                ref_pic_list_temp0[r_idx as usize] = self.rps.ref_pic_set_lt_curr[i]
                    .clone()
                    .map(|e| RefPicListEntry::DpbEntry(e));

                i += 1;
                r_idx += 1;
            }

            if pps.scc_extension().curr_pic_ref_enabled_flag() {
                ref_pic_list_temp0[r_idx as usize] =
                    Some(RefPicListEntry::CurrentPicture(cur_pic.clone()));

                r_idx += 1;
            }
        }

        // Equation 8-9
        for r_idx in 0..=usize::from(hdr.num_ref_idx_l0_active_minus1()) {
            let entry = if rplm.ref_pic_list_modification_flag_l0() {
                let idx = rplm.list_entry_l0()[r_idx];
                ref_pic_list_temp0[idx as usize].clone()
            } else {
                ref_pic_list_temp0[r_idx].clone()
            };

            self.ref_pic_list0[r_idx] = entry;
        }

        if pps.scc_extension().curr_pic_ref_enabled_flag()
            && !rplm.ref_pic_list_modification_flag_l0()
            && num_rps_curr_temp_list0 > (u32::from(hdr.num_ref_idx_l0_active_minus1()) + 1)
        {
            self.ref_pic_list0[r_idx as usize] =
                Some(RefPicListEntry::CurrentPicture(cur_pic.clone()));
        }

        if hdr.type_().is_b() {
            let mut ref_pic_list_temp1: [Option<RefPicListEntry<T>>; MAX_DPB_SIZE] =
                Default::default();

            let num_rps_curr_temp_list1 = std::cmp::max(
                u32::from(hdr.num_ref_idx_l1_active_minus1()) + 1,
                hdr.num_pic_total_curr(),
            );

            // Equation 8-10
            let mut r_idx = 0;
            assert!(num_rps_curr_temp_list1 as usize <= MAX_DPB_SIZE);
            while r_idx < num_rps_curr_temp_list1 {
                let mut i = 0;
                while i < self.rps.num_poc_st_curr_after && r_idx < num_rps_curr_temp_list1 {
                    ref_pic_list_temp1[r_idx as usize] = self.rps.ref_pic_set_st_curr_after[i]
                        .clone()
                        .map(|e| RefPicListEntry::DpbEntry(e));
                    i += 1;
                    r_idx += 1;
                }

                let mut i = 0;
                while i < self.rps.num_poc_st_curr_before && r_idx < num_rps_curr_temp_list1 {
                    ref_pic_list_temp1[r_idx as usize] = self.rps.ref_pic_set_st_curr_before[i]
                        .clone()
                        .map(|e| RefPicListEntry::DpbEntry(e));
                    i += 1;
                    r_idx += 1;
                }

                let mut i = 0;
                while i < self.rps.num_poc_lt_curr && r_idx < num_rps_curr_temp_list1 {
                    ref_pic_list_temp1[r_idx as usize] = self.rps.ref_pic_set_lt_curr[i]
                        .clone()
                        .map(|e| RefPicListEntry::DpbEntry(e));
                    i += 1;
                    r_idx += 1;
                }

                if pps.scc_extension().curr_pic_ref_enabled_flag() {
                    ref_pic_list_temp1[r_idx as usize] =
                        Some(RefPicListEntry::CurrentPicture(cur_pic.clone()));

                    r_idx += 1;
                }
            }

            // Equation 8-11
            for r_idx in 0..=usize::from(hdr.num_ref_idx_l1_active_minus1()) {
                let entry = if rplm.ref_pic_list_modification_flag_l1() {
                    let idx = rplm.list_entry_l1()[r_idx];
                    ref_pic_list_temp1[idx as usize].clone()
                } else {
                    ref_pic_list_temp1[r_idx].clone()
                };

                self.ref_pic_list1[r_idx] = entry;
            }
        }

        Ok(())
    }

    /// Drain the decoder, processing all pending frames.
    fn drain(&mut self) {
        log::debug!("Draining the decoder");

        let pics = self.dpb.drain();

        log::debug!(
            "Adding POCs {:?} to the ready queue while draining",
            pics.iter()
                .map(|p| p.0.borrow().pic_order_cnt_val)
                .collect::<Vec<_>>()
        );

        log::trace!(
            "{:#?}",
            pics.iter().map(|p| p.0.borrow()).collect::<Vec<_>>()
        );

        self.ready_queue.extend(pics.into_iter().map(|h| h.1));
        self.dpb.clear();
    }

    fn clear_ref_lists(&mut self) {
        self.ref_pic_list0 = Default::default();
        self.ref_pic_list1 = Default::default();
        self.rps.ref_pic_set_lt_curr = Default::default();
        self.rps.ref_pic_set_st_curr_after = Default::default();
        self.rps.ref_pic_set_st_curr_before = Default::default();

        self.rps.num_poc_lt_curr = Default::default();
        self.rps.num_poc_lt_foll = Default::default();
        self.rps.num_poc_st_curr_after = Default::default();
        self.rps.num_poc_st_curr_before = Default::default();
        self.rps.num_poc_st_foll = Default::default();
    }

    /// Bumps the DPB if needed.
    fn bump_as_needed(&mut self, bumping_type: BumpingType) -> anyhow::Result<Vec<DpbEntry<T>>> {
        let mut pics = vec![];

        let sps = self
            .parser
            .get_sps(self.cur_sps_id)
            .context("Invalid SPS in bump_as_needed")?;

        let needs_bumping = match bumping_type {
            BumpingType::BeforeDecoding => Dpb::<T>::needs_bumping,
            BumpingType::AfterDecoding => Dpb::<T>::needs_additional_bumping,
        };

        while needs_bumping(&mut self.dpb, sps) {
            match self.dpb.bump(false) {
                Some(pic) => pics.push(pic),
                None => return Ok(pics),
            }
        }

        Ok(pics)
    }

    // See C.5.2.2
    fn update_dpb_before_decoding(&mut self) -> anyhow::Result<()> {
        let cur_pic = self.cur_pic.as_ref().unwrap();

        if cur_pic.is_irap && cur_pic.no_rasl_output_flag && !self.first_picture_after_eos {
            if cur_pic.no_output_of_prior_pics_flag {
                self.dpb.clear();
            } else {
                self.drain();
            }
        } else {
            self.dpb.remove_unused();
            let bumped = self.bump_as_needed(BumpingType::BeforeDecoding)?;

            log::debug!(
                "Adding POCs {:?} to the ready queue before decoding",
                bumped
                    .iter()
                    .map(|p| p.0.borrow().pic_order_cnt_val)
                    .collect::<Vec<_>>()
            );

            log::trace!(
                "{:#?}",
                bumped.iter().map(|p| p.0.borrow()).collect::<Vec<_>>()
            );

            let bumped = bumped.into_iter().map(|p| p.1).collect::<Vec<_>>();
            self.ready_queue.extend(bumped);
        }

        Ok(())
    }

    /// Handle a picture. Called only once per frame.
    fn handle_picture(&mut self, timestamp: u64, slice: &Slice<&[u8]>) -> anyhow::Result<()> {
        let prev_tid0 = self.prev_tid_0_pic.as_ref();

        let pps = self
            .parser
            .get_pps(slice.header().pic_parameter_set_id())
            .context("Invalid PPS in handle_picture")?;

        let _sps = self
            .parser
            .get_sps(pps.seq_parameter_set_id())
            .context("Invalid SPS in handle_picture")?;

        self.cur_pps_id = pps.pic_parameter_set_id();
        self.cur_sps_id = pps.seq_parameter_set_id();
        self.cur_pic = Some(PictureData::new_from_slice(
            slice,
            pps,
            self.first_picture_in_bitstream,
            self.first_picture_after_eos,
            prev_tid0,
            self.max_pic_order_cnt_lsb,
            timestamp,
        ));

        self.first_picture_after_eos = false;
        self.first_picture_in_bitstream = false;

        let cur_pic = self.cur_pic.as_ref().unwrap();

        if cur_pic.is_irap {
            self.irap_no_rasl_output_flag = cur_pic.no_rasl_output_flag;
        } else if cur_pic.nalu_type.is_rasl() && self.irap_no_rasl_output_flag {
            // NOTE – All RASL pictures are leading pictures of an associated
            // BLA or CRA picture. When the associated IRAP picture has
            // NoRaslOutputFlag equal to 1, the RASL picture is not output and
            // may not be correctly decodable, as the RASL picture may contain
            // references to pictures that are not present in the bitstream.
            // RASL pictures are not used as reference pictures for the decoding
            // process of non-RASL pictures.
            log::debug!(
                "Dropping POC {}, as it may not be decodable according to the specification",
                cur_pic.pic_order_cnt_val
            );

            self.cur_pic = None;
            return Ok(());
        }

        log::debug!("Decode picture POC {}", cur_pic.pic_order_cnt_val);

        self.decode_rps(slice)?;
        self.update_dpb_before_decoding()?;

        let cur_pic = self.cur_pic.as_ref().unwrap();

        let mut cur_backend_pic = self.backend.new_picture(cur_pic, timestamp)?;

        self.backend.handle_picture(
            &mut cur_backend_pic,
            cur_pic,
            self.parser
                .get_sps(self.cur_sps_id)
                .context("Invalid SPS in handle_picture")?,
            self.parser
                .get_pps(self.cur_pps_id)
                .context("Invalid PPS in handle_picture")?,
            &self.dpb,
            &self.rps,
            slice,
        )?;

        self.cur_backend_pic = Some(cur_backend_pic);
        Ok(())
    }

    /// Handle a slice. Called once per slice NALU.
    fn handle_slice(&mut self, _timestamp: u64, slice: &Slice<&[u8]>) -> anyhow::Result<()> {
        self.build_ref_pic_lists(slice.header())?;

        self.backend.decode_slice(
            self.cur_backend_pic.as_mut().unwrap(),
            slice,
            self.parser
                .get_sps(self.cur_sps_id)
                .context("Invalid SPS in handle_slice")?,
            self.parser
                .get_pps(self.cur_pps_id)
                .context("Invalid PPS in handle_slice")?,
            &self.dpb,
            &self.ref_pic_list0,
            &self.ref_pic_list1,
        )?;

        Ok(())
    }

    fn finish_picture(&mut self, pic: PictureData, handle: T) -> anyhow::Result<()> {
        log::debug!("Finishing picture POC {:?}", pic.pic_order_cnt_val);

        // 8.3.1
        if pic.valid_for_prev_tid0_pic {
            self.prev_tid_0_pic = Some(pic.clone());
        }

        self.clear_ref_lists();

        // First store the current picture in the DPB, only then we should
        // decide whether to bump.
        self.dpb.store_picture(Rc::new(RefCell::new(pic)), handle)?;
        let bumped = self.bump_as_needed(BumpingType::AfterDecoding)?;

        log::debug!(
            "Adding POCs {:?} to the ready queue after decoding",
            bumped
                .iter()
                .map(|p| p.0.borrow().pic_order_cnt_val)
                .collect::<Vec<_>>()
        );

        log::trace!(
            "{:#?}",
            bumped.iter().map(|p| p.0.borrow()).collect::<Vec<_>>()
        );

        let bumped = bumped.into_iter().map(|p| p.1).collect::<Vec<_>>();
        self.ready_queue.extend(bumped);

        Ok(())
    }

    fn decode_access_unit(
        &mut self,
        timestamp: u64,
        bitstream: &[u8],
    ) -> Result<usize, DecodeError> {
        if self.backend.surface_pool().num_free_surfaces() == 0 {
            return Err(DecodeError::NotEnoughOutputBuffers(1));
        }

        let mut cursor = Cursor::new(bitstream);

        while let Ok(nalu) = Nalu::next(&mut cursor) {
            match nalu.header().type_() {
                NaluType::VpsNut => {
                    self.parser.parse_vps(&nalu)?;
                }
                NaluType::SpsNut => {
                    let sps = self.parser.parse_sps(&nalu)?;
                    self.max_pic_order_cnt_lsb = 1 << (sps.log2_max_pic_order_cnt_lsb_minus4() + 4);
                }

                NaluType::PpsNut => {
                    self.parser.parse_pps(&nalu)?;
                }

                NaluType::BlaWLp
                | NaluType::BlaWRadl
                | NaluType::BlaNLp
                | NaluType::IdrWRadl
                | NaluType::IdrNLp
                | NaluType::TrailN
                | NaluType::TrailR
                | NaluType::TsaN
                | NaluType::TsaR
                | NaluType::StsaN
                | NaluType::StsaR
                | NaluType::RadlN
                | NaluType::RadlR
                | NaluType::RaslN
                | NaluType::RaslR
                | NaluType::CraNut => {
                    let mut slice = self.parser.parse_slice_header(nalu)?;

                    let first_slice_segment_in_pic_flag =
                        slice.header().first_slice_segment_in_pic_flag();

                    if slice.header().dependent_slice_segment_flag() {
                        let previous_independent_header = self.last_independent_slice_header.as_ref().ok_or(anyhow!("Cannot process an dependent slice without first processing and independent one"))?.clone();
                        slice.replace_header(previous_independent_header)?;

                        let pps = self
                            .parser
                            .get_pps(slice.header().pic_parameter_set_id())
                            .context("Invalid SPS when processing a dependent slice")?;

                        let sps = self
                            .parser
                            .get_sps(pps.seq_parameter_set_id())
                            .context("Invalid PPS when processing a dependent slice")?;

                        // A dependent slice may refer to a previous SPS which
                        // is not the one currently in use.
                        if Self::negotiation_possible(sps, &self.dpb, &self.negotiation_info) {
                            self.backend.new_sequence(sps)?;
                            self.decoding_state = DecodingState::AwaitingFormat(sps.clone());
                            continue;
                        }
                    } else {
                        self.last_independent_slice_header = Some(slice.header().clone());
                    }

                    if self.cur_pic.is_none() {
                        if self.backend.surface_pool().num_free_surfaces() == 0 {
                            return Err(DecodeError::NotEnoughOutputBuffers(1));
                        }

                        self.handle_picture(timestamp, &slice)?;
                    } else if first_slice_segment_in_pic_flag {
                        // We have identified a new picture and must submit the
                        // old one before proceeding.
                        let (picture, handle) = self.submit_picture()?;
                        self.finish_picture(picture, handle)?;

                        // TODO: the right place to check is before starting a
                        // picture, all other operations do not consume
                        // resources from the backend. Maybe we should backport
                        // this idea to h.264, as it will also crash if enough
                        // slices make this path be taken too much, which means
                        // that the client would not receive
                        // DecodeError::CheckEvents (in h264's implementation),
                        // but rather "OutOfResources" from the backend, which
                        // will abort ccdec.
                        if self.backend.surface_pool().num_free_surfaces() == 0 {
                            return Err(DecodeError::NotEnoughOutputBuffers(1));
                        }

                        self.handle_picture(timestamp, &slice)?;
                    }

                    if self.cur_pic.is_some() {
                        // Picture may have been dropped during handle_picture()
                        self.handle_slice(timestamp, &slice)?;
                    }
                }

                NaluType::EosNut => {
                    self.first_picture_after_eos = true;
                }

                NaluType::EobNut => {
                    self.first_picture_in_bitstream = true;
                }

                other => {
                    log::debug!("Unsupported NAL unit type {:?}", other,);
                }
            }
        }

        // TODO: this should be refactored as per #facd6fa
        if self.cur_pic.is_some() {
            let (picture, handle) = self.submit_picture()?;
            self.finish_picture(picture, handle)?;
        }

        Ok(bitstream.len())
    }

    /// Submits the picture to the accelerator.
    fn submit_picture(&mut self) -> Result<(PictureData, T), DecodeError> {
        let picture = self.cur_pic.take().unwrap();

        let handle = self
            .backend
            .submit_picture(self.cur_backend_pic.take().unwrap())?;

        if self.blocking_mode == BlockingMode::Blocking {
            handle.sync()?;
        }

        Ok((picture, handle))
    }
}

impl<T, P> StatelessVideoDecoder<T::Descriptor> for Decoder<T, P>
where
    T: DecodedHandle + Clone + 'static,
{
    fn decode(&mut self, timestamp: u64, mut bitstream: &[u8]) -> Result<usize, DecodeError> {
        // let sps = Self::peek_sps(&mut self.parser, bitstream)?;
        let sps = None;

        if let Some(sps) = sps {
            if Self::negotiation_possible(&sps, &self.dpb, &self.negotiation_info) {
                self.backend.new_sequence(&sps)?;
                self.decoding_state = DecodingState::AwaitingFormat(sps);
            } else if matches!(self.decoding_state, DecodingState::Reset) {
                // We can resume decoding since the decoding parameters have not changed.
                self.decoding_state = DecodingState::Decoding;
            }
        } else if matches!(self.decoding_state, DecodingState::Reset) {
            let mut cursor = Cursor::new(bitstream);

            while let Ok(nalu) = Nalu::next(&mut cursor) {
                // In the Reset state we can resume decoding from any key frame.
                if nalu.header().type_().is_idr() {
                    bitstream = &bitstream[nalu.sc_offset()..];
                    self.decoding_state = DecodingState::Decoding;
                    break;
                }
            }
        }

        match &mut self.decoding_state {
            // Skip input until we get information from the stream.
            DecodingState::AwaitingStreamInfo | DecodingState::Reset => Ok(bitstream.len()),
            // Ask the client to confirm the format before we can process this.
            DecodingState::AwaitingFormat(_) => Err(DecodeError::CheckEvents),
            DecodingState::Decoding => self.decode_access_unit(timestamp, bitstream),
        }
    }

    fn flush(&mut self) -> Result<(), DecodeError> {
        self.drain();
        self.decoding_state = DecodingState::Reset;

        Ok(())
    }

    fn next_event(&mut self) -> Option<DecoderEvent<T::Descriptor>> {
        // The next event is either the next frame, or, if we are awaiting negotiation, the format
        // change event that will allow us to keep going.
        (&mut self.ready_queue)
            .next()
            .map(|handle| DecoderEvent::FrameReady(Box::new(handle)))
            .or_else(|| {
                if let DecodingState::AwaitingFormat(sps) = &self.decoding_state {
                    Some(DecoderEvent::FormatChanged(Box::new(
                        StatelessDecoderFormatNegotiator::new(self, sps.clone(), |decoder, sps| {
                            // Apply the SPS settings to the decoder so we don't enter the AwaitingFormat state
                            // on the next decode() call.
                            decoder.apply_sps(sps);
                            decoder.decoding_state = DecodingState::Decoding;
                        }),
                    )))
                } else {
                    None
                }
            })
    }

    fn surface_pool(&mut self) -> &mut dyn SurfacePool<T::Descriptor> {
        self.backend.surface_pool()
    }

    fn stream_info(&self) -> Option<&StreamInfo> {
        self.backend.stream_info()
    }
}

impl<T, P> private::StatelessVideoDecoder for Decoder<T, P>
where
    T: DecodedHandle + Clone,
{
    fn try_format(&mut self, format: crate::DecodedFormat) -> anyhow::Result<()> {
        match &self.decoding_state {
            DecodingState::AwaitingFormat(sps) => self.backend.try_format(sps, format),
            _ => Err(anyhow!(
                "current decoder state does not allow format change"
            )),
        }
    }
}

#[cfg(test)]
pub mod tests {

    use crate::decoder::stateless::h265::Decoder;
    use crate::decoder::stateless::tests::test_decode_stream;
    use crate::decoder::stateless::tests::TestStream;
    use crate::decoder::BlockingMode;
    use crate::utils::simple_playback_loop;
    use crate::utils::simple_playback_loop_owned_surfaces;
    use crate::utils::H265FrameIterator;
    use crate::DecodedFormat;

    /// Run `test` using the dummy decoder, in both blocking and non-blocking modes.
    fn test_decoder_dummy(test: &TestStream, blocking_mode: BlockingMode) {
        let decoder = Decoder::new_dummy(blocking_mode).unwrap();

        test_decode_stream(
            |d, s, f| {
                simple_playback_loop(
                    d,
                    H265FrameIterator::new(s),
                    f,
                    &mut simple_playback_loop_owned_surfaces,
                    DecodedFormat::NV12,
                    blocking_mode,
                )
            },
            decoder,
            test,
            false,
            false,
        );
    }

    /// A 64x64 progressive byte-stream encoded I-frame to make it easier to
    /// spot errors on the libva trace.
    /// Encoded with the following GStreamer pipeline:
    ///
    /// gst-launch-1.0 videotestsrc num-buffers=1 ! video/x-raw,format=I420,width=64,height=64 ! x265enc ! video/x-h265,profile=main ! filesink location="64x64-I.h265"
    pub const DECODE_64X64_PROGRESSIVE_I: TestStream = TestStream {
        stream: include_bytes!("../../codec/h265/test_data/64x64-I.h265"),
        crcs: include_str!("../../codec/h265/test_data/64x64-I.h265.crc"),
    };

    #[test]
    fn test_64x64_progressive_i_block() {
        test_decoder_dummy(&DECODE_64X64_PROGRESSIVE_I, BlockingMode::Blocking);
    }

    #[test]
    fn test_64x64_progressive_i_nonblock() {
        test_decoder_dummy(&DECODE_64X64_PROGRESSIVE_I, BlockingMode::NonBlocking);
    }

    /// A 64x64 progressive byte-stream encoded I-frame and P-frame to make
    /// it easier to spot errors on the libva trace.
    /// Encoded with the following GStreamer pipeline:
    /// gst-launch-1.0 videotestsrc num-buffers=2 ! video/x-raw,format=I420,width=64,height=64 ! x265enc option-string="b-adapt=0" ! video/x-h265,profile=main ! filesink location="64x64-I-P.h265"
    pub const DECODE_64X64_PROGRESSIVE_I_P: TestStream = TestStream {
        stream: include_bytes!("../../codec/h265/test_data/64x64-I-P.h265"),
        crcs: include_str!("../../codec/h265/test_data/64x64-I-P.h265.crc"),
    };

    #[test]
    fn test_64x64_progressive_i_p_block() {
        test_decoder_dummy(&DECODE_64X64_PROGRESSIVE_I_P, BlockingMode::Blocking);
    }

    #[test]
    fn test_64x64_progressive_i_p_nonblock() {
        test_decoder_dummy(&DECODE_64X64_PROGRESSIVE_I_P, BlockingMode::NonBlocking);
    }

    /// A 64x64 progressive byte-stream encoded I-P-B-P sequence to make it
    /// easier to it easier to spot errors on the libva trace.
    /// Encoded with the following GStreamer pipeline:
    /// gst-launch-1.0 videotestsrc num-buffers=3 ! video/x-raw,format=I420,width=64,height=64 ! x265enc option-string="b-adapt=0:bframes=1" ! video/x-h265,profile=main ! filesink location="64x64-I-P-B-P.h265"
    pub const DECODE_64X64_PROGRESSIVE_I_P_B_P: TestStream = TestStream {
        stream: include_bytes!("../../codec/h265/test_data/64x64-I-P-B-P.h265"),
        crcs: include_str!("../../codec/h265/test_data/64x64-I-P-B-P.h265.crc"),
    };

    #[test]
    fn test_64x64_progressive_i_p_b_p_block() {
        test_decoder_dummy(&DECODE_64X64_PROGRESSIVE_I_P_B_P, BlockingMode::Blocking);
    }

    #[test]
    fn test_64x64_progressive_i_p_b_p_nonblock() {
        test_decoder_dummy(&DECODE_64X64_PROGRESSIVE_I_P_B_P, BlockingMode::NonBlocking);
    }

    /// Same as Chromium's test-25fps.h265
    pub const DECODE_TEST_25FPS: TestStream = TestStream {
        stream: include_bytes!("../../codec/h265/test_data/test-25fps.h265"),
        crcs: include_str!("../../codec/h265/test_data/test-25fps.h265.crc"),
    };

    #[test]
    fn test_25fps_block() {
        test_decoder_dummy(&DECODE_TEST_25FPS, BlockingMode::Blocking);
    }

    #[test]
    fn test_25fps_nonblock() {
        test_decoder_dummy(&DECODE_TEST_25FPS, BlockingMode::NonBlocking);
    }

    /// Same as Chromium's bear.h265
    pub const DECODE_BEAR: TestStream = TestStream {
        stream: include_bytes!("../../codec/h265/test_data/bear.h265"),
        crcs: include_str!("../../codec/h265/test_data/bear.h265.crc"),
    };

    #[test]
    fn test_bear_block() {
        test_decoder_dummy(&DECODE_BEAR, BlockingMode::Blocking);
    }

    #[test]
    fn test_bear_nonblock() {
        test_decoder_dummy(&DECODE_BEAR, BlockingMode::NonBlocking);
    }

    /// Same as Chromium's bbb.h265
    pub const DECODE_BBB: TestStream = TestStream {
        stream: include_bytes!("../../codec/h265/test_data/bbb.h265"),
        crcs: include_str!("../../codec/h265/test_data/bbb.h265.crc"),
    };

    #[test]
    fn test_bbb_block() {
        test_decoder_dummy(&DECODE_BBB, BlockingMode::Blocking);
    }

    #[test]
    fn test_bbb_nonblock() {
        test_decoder_dummy(&DECODE_BBB, BlockingMode::NonBlocking);
    }
}
