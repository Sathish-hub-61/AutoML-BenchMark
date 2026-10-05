class DatasetError(Exception):
    """Base error for dataset loading and validation."""


class TargetNotFoundError(DatasetError):
    pass


class InsufficientRowsError(DatasetError):
    pass


class SingleClassError(DatasetError):
    pass


class AllNullColumnError(DatasetError):
    pass


class DuplicateColumnError(DatasetError):
    pass


class RegressionLikeTargetError(DatasetError):
    pass


class UnsupportedSourceError(DatasetError):
    pass
