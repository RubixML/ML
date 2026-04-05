<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\DataProvider;

final class GradientBoostProvider
{
    /**
     * Return sample dataset sizes for additional GradientBoost tests.
     *
     * @return array<string, array{0: int, 1: int}>
     */
    public static function trainPredictAdditionalProvider() : array
    {
        return [
            'default swiss roll sample' => [512, 256],
            'smaller swiss roll sample' => [128, 64],
        ];
    }
}
