<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\DataProvider;

use Generator;

final class RegressionTreeProvider
{
    /**
     * Return dataset sizes for additional RegressionTree tests.
     *
     * @return Generator<string, array{0: int, 1: int}>
     */
    public static function trainedModelCases() : Generator
    {
        yield 'standard split' => [512, 256];

        yield 'smaller split' => [128, 64];
    }
}
