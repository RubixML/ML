<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\DataProvider;

use Generator;

final class GradientBoostProvider
{
    /**
     * Return sample dataset sizes for additional GradientBoost tests.
     *
     * @return Generator<string, array{0: int, 1: int}>
     */
    public static function trainPredictAdditionalProvider() : Generator
    {
        yield 'default swiss roll sample' => [512, 256];

        yield 'smaller swiss roll sample' => [128, 64];
    }
}
