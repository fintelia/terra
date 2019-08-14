use gfx_hal::buffer::Usage;
use rendy::factory::Factory;
use gfx_hal::Backend;
use std::sync::mpsc::{self, Sender, Receiver};
use std::collections::VecDeque;

const BUFFER_SIZE: u64 = 64 << 20;

pub struct Fence(Arc<AtomicBool>);

struct Upload<B: Backend> {
    image: Handle<Image<B>>,
    data: Vec<u8>,
    layer: usize,
    fence: Fence,
}

pub struct Uploader<B: Backend> {
    receiver: Receiver<Upload<B>>,

    pending: VecDeque<Upload>,
}
impl<B: Backend> Uploader<B: Backend> {
    pub fn new(factory: &mut Factory) -> (Self, Sender<Upload>) {
        let (sender, receiver) = mpsc::channel();
        let buffer = factory.create_buffer(BufferInfo {
            size: BUFFER_SIZE,
            usage: Usage::TRANSFER_SRC | Usage::TRANSFER_DST,
        }, Upload);

        (
            Self { receiver, pending: VecDeque::new(), buffer,  },
            sender
        )
    }

    pub fn upload()
    
    /// Returns whether there are more uploads pending.
    pub fn run(&mut self, device: &Device<B>) -> bool {
        
        true
    }
}
