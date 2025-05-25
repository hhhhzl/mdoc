"""
"""
import heapq
from abc import ABC, abstractmethod

# Project includes.


class PriorityQueue(ABC):
    """
    An abstract class for defining a priority queue.
    """
    def __init__(self):
        pass

    @abstractmethod
    def push(self, item):
        pass

    @abstractmethod
    def pop(self):
        pass

    @abstractmethod
    def size(self) -> int:
        pass

    @abstractmethod
    def clear(self):
        pass


class FocalQueue(PriorityQueue):
    """
    A focal queue implementation.
    It keeps two a few lists:
    - waitlist: A list of items that are waiting to be pushed to the queue.
    - focal: The actual queue.
    - open: The anchor of the focal queue. This contains all elements. In FOCAL and not.

    The items added here should all have the get_cost, get_id, and get_subcost methods implemented.
    Cost is used for open, and subcost is used for focal.
    """
    def __init__(self, w=1.3):
        super().__init__()
        # Contains all elements that are with f < f_min * w .
        self.focal = []
        # Contains all elements that were pushed but not yet added to focal.
        self.waitlist = []

        # Some parameters.
        self.w = w

        # Some elements to keep track of.
        self.f_min_open = float('inf')

    def push(self, elem):
        heapq.heappush(self.waitlist, (elem.get_cost(), elem.get_id(), elem))
        self.f_min_open = min(self.f_min_open, elem.get_cost())

    def update_f_min_open(self):
        if len(self.waitlist) > 0:
            self.f_min_open = self.waitlist[0][0]
        if len(self.focal) > 0:
            self.f_min_open = min(self.f_min_open, self.focal[0][1])

    def pop(self):
        # Update the focal list by iterating through the waitlist and adding all elements with f < f_min * w.
        self.update_f_min_open()

        f_max = self.f_min_open * self.w
        while len(self.waitlist) > 0:
            elem = heapq.heappop(self.waitlist)
            if elem[0] <= f_max:
                # Push as subcost, cost, id, elem.
                heapq.heappush(self.focal, (elem[2].get_subcost(), elem[0], elem[1], elem[2]))
            else:
                # If not in bound, push back to waitlist.
                heapq.heappush(self.waitlist, elem)
                break

        # Return the element with the lowest subcost.
        elem_to_return = heapq.heappop(self.focal)[-1]
        return elem_to_return

    def size(self) -> int:
        return len(self.focal) + len(self.waitlist)

    def __len__(self):
        return self.size()

    def clear(self):
        self.focal = []
        self.waitlist = []