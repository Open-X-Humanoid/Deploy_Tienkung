import queue

class PeekableQueue(queue.Queue):
    def peek(self):
        """取出再放回"""
        item = self.get_nowait()
        try:
            self.put_nowait(item)
        except queue.Full:
            pass
        return item