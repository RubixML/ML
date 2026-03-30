<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\DataProvider;

final class AdalineProvider
{
    /**
     * Return the shared training samples for Adaline sample-based tests.
     *
     * @return array<string, array{0: list<list<int>>, 1: list<int>, 2: list<int>}>
     */
    public static function trainPredictProvider() : array
    {
        return [
            '1 feature linear sample' => [
                [
                    [0],
                    [1],
                    [2],
                    [3],
                ],
                [3, 5, 7, 9],
                [4],
            ],
            '2 feature linear sample' => [
                [
                    [0, 0],
                    [1, 1],
                    [2, 1],
                    [1, 2],
                ],
                [3, 6, 7, 8],
                [2, 2],
            ],
            '3 feature linear sample' => [
                [
                    [0, 0, 0],
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                ],
                [4, 5, 6, 7],
                [1, 1, 1],
            ],
        ];
    }
}
