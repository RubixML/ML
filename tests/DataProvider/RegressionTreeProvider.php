<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\DataProvider;

final class RegressionTreeProvider
{
    /**
     * Return dataset sizes for additional RegressionTree tests.
     *
     * @return array<string, array{0: int, 1: int}>
     */
    public static function trainedModelCases() : array
    {
        return [
            'standard split' => [512, 256],
            'smaller split' => [128, 64],
        ];
    }
}
