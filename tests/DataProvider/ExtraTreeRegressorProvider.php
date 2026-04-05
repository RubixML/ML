<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\DataProvider;

use Generator;

final class ExtraTreeRegressorProvider
{
    /**
     * Return sample datasets for additional ExtraTreeRegressor tests.
     *
     * @return Generator<string, array{0: list<list<int>>, 1: list<int>, 2: list<int>}>
     */
    public static function trainPredictProvider() : Generator
    {
        yield '1 feature sample' => [
                [
                    [0],
                    [1],
                    [2],
                    [3],
                ],
                [2, 4, 6, 8],
                [4],
        ];

        yield '2 feature sample' => [
                [
                    [0, 0],
                    [1, 1],
                    [2, 1],
                    [1, 2],
                ],
                [3, 6, 7, 8],
                [2, 2],
        ];

        yield '3 feature sample' => [
                [
                    [0, 0, 0],
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                ],
                [4, 5, 6, 7],
                [1, 1, 1],
        ];

        yield '4 feature sample' => [
                [
                    [0, 0, 0, 0],
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                ],
                [2, 4, 6, 8],
                [1, 1, 1, 1],
        ];
    }
}
