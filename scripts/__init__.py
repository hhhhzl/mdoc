from enum import Enum


class EnvironmentType(Enum):
    # Single tile.
    EMPTY_DISK_CIRCLE = "EnvEmpty2DRobotPlanarDiskCircle"
    EMPTY_DISK_BOUNDARY = "EnvEmpty2DRobotPlanarDiskBoundary"
    EMPTY_DISK_RANDOM = "EnvEmpty2DRobotPlanarDiskRandom"

    CONVEYOR_DISK_BOUNDARY = "EnvConveyor2DRobotPlanarDiskBoundary"
    CONVEYOR_DISK_RANDOM = "EnvConveyor2DRobotPlanarDiskRandom"

    HIGHWAYS_DISK_SMALL_CIRCLE = "EnvHighways2DRobotPlanarDiskSmallCircle"
    HIGHWAYS_DISK_RANDOM = "EnvHighways2DRobotPlanarDiskRandom"
    DROP_REGION_DISK_BOUNDARY = "EnvDropRegion2DRobotPlanarDiskBoundary"
    # Multiple tiles.
    TEST_2X2_RANDOM = "EnvTestTwoByTwoRobotPlanarDiskRandom"
    TEST_3X3_RANDOM = "EnvTestThreeByThreeRobotPlanarDiskRandom"
    TEST_4X4_RANDOM = "EnvTestFourByFourRobotPlanarDiskRandom"

    @classmethod
    def choices(cls):
        return [e.value for e in cls]

    @classmethod
    def from_string(cls, s):
        for e in cls:
            if e.value == s:
                return e
        raise ValueError(f"Invalid environment name: {s}. Valid options: {cls.choices()}")


class MultiAgentPlannerType(Enum):
    PP = "PP"
    CBS = "CBS"
    ECBS = "ECBS"
    XECBS = "XECBS"
    XCBS = "XCBS"

    @classmethod
    def choices(cls):
        return [e.value for e in cls]


class LowerPlannerMethodType(Enum):
    MDOC = 'MDOCEnsemble'

    # baselines
    WASTAR = 'WASTAR'
    KCBS = "KCBSLower"
    MMD = 'MMDEnsemble'


    @classmethod
    def choices(cls):
        return [e.value for e in cls]