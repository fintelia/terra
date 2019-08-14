
use std::collections::{HashMap, VecDequeue};

pub struct ArrayContents<K, V> {
    forward: HashMap<K, (V, usize)>,
    reverse: Box<[Option<K>]>,
    free: Vec<usize>,
    used: VecDeque<usize>,
}
impl<K: Copy, V> ArrayContainer<K, V> {
    fn new(size: usize) -> Self {
        Self {
            forward: HashMap::new(),
            reverse: vec![None; size].into_boxed_slice(),
            free: (0..size).collect(),
            used: VecDeque::new(),
        }
    }

    fn insert(&mut self, k: K, v: V) -> usize {
        let i = if self.free.pop().or_else(|| self.used.pop_front());

        self.reverse[i] = Some(K);
        used.push(i);
    }
}
