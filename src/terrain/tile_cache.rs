use vec_map::VecMap;
use terrain::quadtree::{Node, NodeId};

#[derive(Clone, Copy, PartialEq, PartialOrd)]
pub struct Priority(f32);
impl Priority {
    pub fn cutoff() -> Self {
        Priority(1.0)
    }
    pub fn none() -> Self {
        Priority(-1.0)
    }
    pub fn from_f32(value: f32) -> Self {
        assert!(value.is_finite());
        Priority(value)
    }
}
impl Eq for Priority {}
impl Ord for Priority {
    fn cmp(&self, other: &Self) -> ::std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

pub(crate) struct TileCache {
    /// Maximum number of slots in this `TileCache`.
    size: usize,
    /// Actually contents of the cache.
    slots: Vec<(Priority, NodeId)>,
    /// Which index each node is at in the cache (if any).
    reverse: VecMap<usize>,
    /// Nodes that should be added to the cache.
    missing: Vec<(Priority, NodeId)>,
    /// Smallest priority among all nodes in the cache.
    min_priority: Priority,
}
impl TileCache {
    pub fn new(size: usize) -> Self {
        Self {
            size,
            slots: Vec::new(),
            reverse: VecMap::new(),
            missing: Vec::new(),
            min_priority: Priority::none(),
        }
    }

    pub fn update_priorities(&mut self, nodes: &mut Vec<Node>) {
        for &mut (ref mut priority, id) in self.slots.iter_mut() {
            *priority = nodes[id].priority();
        }

        self.min_priority = self.slots.iter().map(|s| s.0).min().unwrap_or(
            Priority::none(),
        );
    }

    pub fn add_missing(&mut self, element: (Priority, NodeId)) {
        if element.0 > self.min_priority || self.slots.len() < self.size {
            self.missing.push(element);
        }
    }

    pub fn load_missing(&mut self, nodes: &mut Vec<Node>) {
        self.missing.sort();

        while self.slots.len() < self.size {
            if self.missing.is_empty() {
                return;
            }
            let index = self.slots.len();
            let id = self.missing.pop().unwrap().1;
            self.load(id, &mut nodes[id], index);
        }

        let mut index = 0;
        while let Some((priority, id)) = self.missing.pop() {
            while index < self.slots.len() && self.slots[index].0 >= priority {
                index += 1;
            }

            if self.slots[index].0 < priority {
                self.load(id, &mut nodes[id], index);
                index += 1;
            } else {
                // The cache is full and all the tiles in the cache have higher
                // priority than the remaining missing tiles. Give up loading the rest.
                self.missing.clear();
                return;
            }
        }
    }

    fn load(&mut self, id: NodeId, _node: &mut Node, slot: usize) {
        if slot < self.slots.len() {
            self.reverse.remove(self.slots[slot].1.index());
        }
        self.reverse.insert(id.index(), slot);
        unimplemented!()
    }

    pub fn contains(&self, id: NodeId) -> bool {
        self.reverse.contains_key(id.index())
    }
}
