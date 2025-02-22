// Copyright 2022 The ChromiumOS Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

use crate::decoders::vp9::decoder::Segmentation;
use crate::decoders::vp9::parser::Header;
use crate::decoders::vp9::parser::MAX_SEGMENTS;
use crate::decoders::vp9::parser::NUM_REF_FRAMES;
use crate::decoders::VideoDecoderBackend;

#[cfg(test)]
pub mod dummy;
#[cfg(feature = "vaapi")]
pub mod vaapi;

pub type Result<T> = crate::decoders::StatelessBackendResult<T>;

/// Trait for stateless decoder backends. The decoder will call into the backend
/// to request decode operations. The backend can operate in blocking mode,
/// where it will wait until the current decode finishes, or in non-blocking
/// mode, where it should return immediately with any previously decoded frames
/// that happen to be ready.
pub(crate) trait StatelessDecoderBackend: VideoDecoderBackend {
    /// Called when new stream parameters are found.
    fn new_sequence(&mut self, header: &Header) -> Result<()>;

    /// Called when the decoder wants the backend to finish the decoding
    /// operations for `picture`.
    ///
    /// This call will assign the ownership of the BackendHandle to the Picture
    /// and then assign the ownership of the Picture to the Handle.
    fn submit_picture(
        &mut self,
        picture: &Header,
        reference_frames: &[Option<Self::Handle>; NUM_REF_FRAMES],
        bitstream: &[u8],
        timestamp: u64,
        segmentation: &[Segmentation; MAX_SEGMENTS],
    ) -> Result<Self::Handle>;
}
