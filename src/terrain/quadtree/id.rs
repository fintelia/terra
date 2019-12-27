use crate::terrain::quadtree::Node;
use serde::{Deserialize, Serialize};
use std::num::NonZeroU32;
use std::ops::{Index, IndexMut};

#[derive(Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct NodeId(NonZeroU32);
impl NodeId {
    pub fn root() -> Self {
        NodeId(NonZeroU32::new(1).unwrap())
    }
    pub fn index(&self) -> usize {
        self.0.get() as usize - 1
    }
    pub fn new(id: u32) -> Self {
        NodeId(NonZeroU32::new(id + 1).unwrap())
    }
}
impl Index<NodeId> for Vec<Node> {
    type Output = Node;
    fn index(&self, id: NodeId) -> &Node {
        &self[id.index()]
    }
}
impl IndexMut<NodeId> for Vec<Node> {
    fn index_mut(&mut self, id: NodeId) -> &mut Node {
        &mut self[id.index()]
    }
}
