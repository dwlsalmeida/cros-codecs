// Copyright 2022 The ChromiumOS Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

//! This file contains a dummy backend whose only purpose is to let the decoder
//! run so we can test it in isolation.

use std::cell::RefCell;
use std::rc::Rc;

use crate::decoder::stateless::StatelessDecoderBackend;
use crate::decoder::DecodedHandle;
use crate::decoder::DynHandle;
use crate::decoder::MappableHandle;
use crate::decoder::StreamInfo;
use crate::decoder::SurfacePool;
use crate::DecodedFormat;
use crate::Resolution;

#[derive(Default)]
pub struct BackendHandle(());

impl MappableHandle for BackendHandle {
    fn read(&mut self, _: &mut [u8]) -> anyhow::Result<()> {
        Ok(())
    }

    fn image_size(&mut self) -> usize {
        1
    }
}

impl DynHandle for BackendHandle {
    fn dyn_mappable_handle<'a>(&'a self) -> anyhow::Result<Box<dyn MappableHandle + 'a>> {
        Ok(Box::<BackendHandle>::default())
    }
}

pub struct Handle {
    pub handle: Rc<RefCell<BackendHandle>>,
}

impl Clone for Handle {
    fn clone(&self) -> Self {
        Self {
            handle: Rc::clone(&self.handle),
        }
    }
}

impl DecodedHandle<()> for Handle {
    fn coded_resolution(&self) -> Resolution {
        Default::default()
    }

    fn display_resolution(&self) -> Resolution {
        Default::default()
    }

    fn timestamp(&self) -> u64 {
        0
    }

    fn dyn_picture(&self) -> std::cell::Ref<dyn DynHandle> {
        self.handle.borrow()
    }

    fn sync(&self) -> anyhow::Result<()> {
        Ok(())
    }

    fn is_ready(&self) -> bool {
        true
    }

    fn resource(&self) -> std::cell::Ref<()> {
        std::cell::Ref::map(self.handle.borrow(), |h| &h.0)
    }
}

/// Dummy backend that can be used for any codec.
pub(crate) struct Backend {
    stream_info: StreamInfo,
}

impl Backend {
    pub(crate) fn new() -> Self {
        Self {
            stream_info: StreamInfo {
                format: DecodedFormat::I420,
                min_num_surfaces: 4,
                coded_resolution: Resolution::from((320, 200)),
                display_resolution: Resolution::from((320, 200)),
            },
        }
    }
}

impl<M> SurfacePool<M> for Backend {
    fn coded_resolution(&self) -> Resolution {
        Resolution::from((320, 200))
    }

    fn set_coded_resolution(&mut self, _resolution: Resolution) {}

    fn add_surfaces(&mut self, _descriptors: Vec<M>) -> Result<(), anyhow::Error> {
        Ok(())
    }

    fn num_free_surfaces(&self) -> usize {
        4
    }

    fn num_managed_surfaces(&self) -> usize {
        4
    }

    fn clear(&mut self) {}
}

impl<FormatInfo> StatelessDecoderBackend<FormatInfo, ()> for Backend {
    type Handle = Handle;

    fn try_format(&mut self, _: &FormatInfo, _: DecodedFormat) -> anyhow::Result<()> {
        Ok(())
    }

    fn stream_info(&self) -> Option<&StreamInfo> {
        Some(&self.stream_info)
    }

    fn surface_pool(&mut self) -> &mut dyn SurfacePool<()> {
        self
    }
}
