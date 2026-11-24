import numpy as np


MAX_DISTANCE = 50
MAX_PATH_LENGTH = 50


class Tracker:
    def __init__(self, maxLost=30, maxDistance=MAX_DISTANCE, maxPathLength=MAX_PATH_LENGTH):
        """
        Initialize the tracker.

        Parameters:
        maxLost (int): Maximum number of frames an object can be lost before deregistration.
        maxDistance (float): Maximum centroid distance to match detections to existing tracks.
        maxPathLength (int): Maximum number of centroids kept for each object's trail.
        """
        self.nextObjectID = 0
        self.objects = {}          # objectID -> current centroid (x, y)
        self.lost = {}             # objectID -> consecutive frames lost
        self.paths = {}            # objectID -> list of centroids
        self.maxLost = maxLost
        self.maxDistance = maxDistance
        self.maxPathLength = maxPathLength

    def register(self, centroid):
        """
        Register a new object with the next available ID.

        Parameters:
        centroid: The centroid of the new object to register (x, y).

        Returns:
        None
        """
        objectID = self.nextObjectID
        self.objects[objectID] = tuple(centroid)
        self.lost[objectID] = 0
        self.paths[objectID] = [tuple(centroid)]
        self.nextObjectID += 1

    def deregister(self, objectID):
        """
        Deregister an object, removing it from tracking.

        Parameters:
        objectID: The ID of the object to deregister.

        Returns:
        None
        """
        if objectID in self.objects:
            del self.objects[objectID]
        if objectID in self.lost:
            del self.lost[objectID]
        if objectID in self.paths:
            del self.paths[objectID]

    def update(self, inputCentroids):
        """
        Update the tracked objects with new centroid information from the current frame.

        Parameters:
        inputCentroids: list of centroids detected in the current frame.

        Returns:
        Updated objects with their current centroid positions.
        """
        inputCentroids = [tuple(c) for c in inputCentroids]

        # No detections this frame: mark everything as lost / deregister if needed
        if len(inputCentroids) == 0:
            for objectID in list(self.lost.keys()):
                self.lost[objectID] += 1
                if self.lost[objectID] > self.maxLost:
                    self.deregister(objectID)
            return self.objects

        # No existing tracks: register every detection
        if len(self.objects) == 0:
            for centroid in inputCentroids:
                self.register(centroid)
            return self.objects

        objectIDs = list(self.objects.keys())
        objectCentroids = list(self.objects.values())

        # Distance matrix: rows = existing tracks, cols = new detections
        D = self._distance(objectCentroids, inputCentroids)

        # Greedy nearest-neighbor matching (naive SORT-style association)
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        usedRows = set()
        usedCols = set()

        for row, col in zip(rows, cols):
            if row in usedRows or col in usedCols:
                continue

            if D[row, col] > self.maxDistance:
                continue

            objectID = objectIDs[row]
            self.objects[objectID] = inputCentroids[col]
            self.lost[objectID] = 0
            self.paths[objectID].append(inputCentroids[col])
            if len(self.paths[objectID]) > self.maxPathLength:
                self.paths[objectID] = self.paths[objectID][-self.maxPathLength:]

            usedRows.add(row)
            usedCols.add(col)

        # Unmatched existing tracks: increment lost count
        unusedRows = set(range(D.shape[0])) - usedRows
        for row in unusedRows:
            objectID = objectIDs[row]
            self.lost[objectID] += 1
            if self.lost[objectID] > self.maxLost:
                self.deregister(objectID)

        # Unmatched detections: register as new objects
        unusedCols = set(range(D.shape[1])) - usedCols
        for col in unusedCols:
            self.register(inputCentroids[col])

        return self.objects

    def get_paths(self):
        """
        Retrieve the paths (list of centroids) of all tracked objects.

        Returns:
        Paths of tracked objects.
        """
        return self.paths

    def _distance(self, a, b):
        """
        Calculate the Euclidean distances between two sets of centroids.

        Parameters:
        a: First set of centroids.
        b: Second set of centroids.

        Returns:
        Matrix of distances between each pair of centroids from set 'a' and 'b'.
        """
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)

        if a.ndim == 1:
            a = a.reshape(1, -1)
        if b.ndim == 1:
            b = b.reshape(1, -1)

        # (N,1,2) - (1,M,2) -> (N,M,2) -> (N,M)
        diff = a[:, None, :] - b[None, :, :]
        return np.sqrt(np.sum(diff ** 2, axis=2))
