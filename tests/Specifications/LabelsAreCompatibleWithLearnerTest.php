<?php

namespace Rubix\ML\Tests\Specifications;

use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Classifiers\AdaBoost;
use Rubix\ML\Regressors\GradientBoost;
use Rubix\ML\Specifications\Specification;
use Rubix\ML\Specifications\LabelsAreCompatibleWithLearner;
use PHPUnit\Framework\TestCase;
use Generator;

/**
 * @group Specifications
 * @covers \Rubix\ML\Specifications\LabelsAreCompatibleWithLearner
 */
class LabelsAreCompatibleWithLearnerTest extends TestCase
{
    /**
     * @test
     * @dataProvider passesProvider
     *
     * @param \Rubix\ML\Specifications\Specification $spec
     * @param bool $expected
     */
    public function passes(Specification $spec, bool $expected) : void
    {
        $this->assertSame($expected, $spec->passes());
    }

    /**
     * @return \Generator<array>
     */
    public function passesProvider() : Generator
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
}
