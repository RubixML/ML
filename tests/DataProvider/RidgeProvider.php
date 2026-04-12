<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\DataProvider;

use Generator;

final class RidgeProvider
{
    /**
     * Return dataset sizes for additional RidgeProvider tests with legacy values.
     *
     * @return Generator<string, array{0: int, 1: int}>
     */
    public static function trainPredictProvider() : Generator
    {
        yield 'sample with 1 feature and smaller values' => [
            [
                [0],
                [1],
                [2],
                [3],
            ],
            [3, 5, 7, 9],
            [4],
            11.0,
            [2.0],
            3.0,
        ];

        yield 'sample with 2 features and smaller values' => [
            [
                [0, 0],
                [1, 1],
                [2, 1],
                [1, 2],
            ],
            [3, 6, 7, 8],
            [2, 2],
            9.0,
            [1.0, 2.0],
            3.0,
        ];

        yield 'sample with 3 features and smaller values' => [
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ],
            [4, 5, 6, 7],
            [1, 1, 1],
            10.0,
            [1.0, 2.0, 3.0],
            4.0,
        ];

        yield 'sample with 4 features' => [
            [
                [50, 3, 5, 10],
                [70, 10, 3, 5],
                [40, 2, 8, 30],
            ],
            [66000, 95000, 45000],
            [60, 5, 4, 12],
            78037.05,
            [1192.98, 401.06, -132.47, -413.58],
            9949.78,
        ];

        yield 'sample with 4 features with shifted values' => [
            [
                [52, 4, 6, 12],
                [71, 9, 4, 6],
                [38, 3, 7, 28],
            ],
            [66000, 95000, 45000],
            [60, 5, 4, 12],
            77709.72,
            [1368.77, 442.49, -158.60, -77.49],
            -5054.98,
        ];
    }

    /**
     * Return dataset sizes for additional RidgeProvider tests with NumPower.
     *
     * @return Generator<string, array{0: int, 1: int}>
     */
    public static function trainPredictProviderForNumPower() : Generator
    {
        $isArm = in_array(strtolower(php_uname('m')), ['arm64', 'aarch64'], true);

        yield 'sample with 1 feature and smaller values' => [
            [
                [0],
                [1],
                [2],
                [3],
            ],
            [3, 5, 7, 9],
            [4],
            11.0,
            [2.0],
            3.0,
        ];

        yield 'sample with 2 features and smaller values' => [
            [
                [0, 0],
                [1, 1],
                [2, 1],
                [1, 2],
            ],
            [3, 6, 7, 8],
            [2, 2],
            9.0,
            [1.0, 2.0],
            3.0,
        ];

        yield 'sample with 3 features and smaller values' => [
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ],
            [4, 5, 6, 7],
            [1, 1, 1],
            10.0,
            [1.0, 2.0, 3.0],
            4.0,
        ];

        yield 'sample with 4 features' => [
            [
                [50, 3, 5, 10],
                [70, 10, 3, 5],
                [40, 2, 8, 30],
            ],
            [66000, 95000, 45000],
            [60, 5, 4, 12],
            $isArm ? 77676.53 : 77644.0,
            $isArm
                ? [1208.26, 360.18, -96.53, -420.41]
                : [1172.0, 452.0, -70.0, -424.0],
            $isArm ? 8810.75 : 10432.0,
        ];

        yield 'sample with 4 features with shifted values' => [
            [
                [52, 4, 6, 12],
                [71, 9, 4, 6],
                [38, 3, 7, 28],
            ],
            [66000, 95000, 45000],
            [60, 5, 4, 12],
            $isArm ? 77585.35 : 78540.0,
            $isArm
                ? [1364.07, 476.45, -161.59, -82.90]
                : [1366.0, 504.0, -156.0, -91.0],
            $isArm ? -4999.93 : -4224.0,
        ];
    }
}
