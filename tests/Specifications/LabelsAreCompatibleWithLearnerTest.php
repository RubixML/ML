<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Specifications;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Classifiers\AdaBoost;
use Rubix\ML\Regressors\GradientBoost;
use Rubix\ML\Specifications\LabelsAreCompatibleWithLearner;
use PHPUnit\Framework\TestCase;
use Generator;

#[Group('Specifications')]
#[CoversClass(LabelsAreCompatibleWithLearner::class)]
class LabelsAreCompatibleWithLearnerTest extends TestCase
{
    public static function passesProvider() : Generator
    {
        yield [
            LabelsAreCompatibleWithLearner::with(
                Labeled::quick([
                    [6.0, -1.1, 5, 'college'],
                ], [200]),
                new GradientBoost()
            ),
            true,
        ];

        yield [
            LabelsAreCompatibleWithLearner::with(
                Labeled::quick([
                    [6.0, -1.1, 5, 'college'],
                ], ['stormy night']),
                new AdaBoost()
            ),
            true,
        ];

        yield [
            LabelsAreCompatibleWithLearner::with(
                Labeled::quick([
                    [6.0, -1.1, 5, 'college'],
                ], ['stormy night']),
                new GradientBoost()
            ),
            false,
        ];

        yield [
            LabelsAreCompatibleWithLearner::with(
                Labeled::quick([
                    [6.0, -1.1, 5, 'college'],
                ], [200]),
                new AdaBoost()
            ),
            false,
        ];
    }

    /**
     * @param LabelsAreCompatibleWithLearner $specification
     * @param bool $expected
     */
    #[DataProvider('passesProvider')]
    public function testPasses(LabelsAreCompatibleWithLearner $specification, bool $expected) : void
    {
        $this->assertSame($expected, $specification->passes());
    }
}
