<?php

namespace Rubix\ML\Tests\Specifications;

use Rubix\ML\Learner;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Classifiers\AdaBoost;
use Rubix\ML\Regressors\GradientBoost;
use Rubix\ML\Specifications\LabelsAreCompatibleWithLearner;
use PHPUnit\Framework\TestCase;
use InvalidArgumentException;
use Generator;

/**
 * @group Specifications
 * @covers \Rubix\ML\Specifications\LabelsAreCompatibleWithLearner
 */
class LabelsAreCompatibleWithLearnerTest extends TestCase
{
    /**
     * @test
     * @dataProvider checkProvider
     *
     * @param \Rubix\ML\Datasets\Labeled $dataset
     * @param \Rubix\ML\Learner $estimator
     * @param bool $valid
     */
    public function check(Labeled $dataset, Learner $estimator, bool $valid) : void
    {
        if (!$valid) {
            $this->expectException(InvalidArgumentException::class);
        }

        LabelsAreCompatibleWithLearner::check($dataset, $estimator);

        $this->assertTrue($valid);
    }

    /**
     * @return \Generator<array>
     */
    public function checkProvider() : Generator
    {
        yield [
            Labeled::quick([
                [6.0, -1.1, 5, 'college'],
            ], [200]),
            new GradientBoost(),
            true,
        ];

        yield [
            Labeled::quick([
                [6.0, -1.1, 5, 'college'],
            ], ['stormy night']),
            new AdaBoost(),
            true,
        ];

        yield [
            Labeled::quick([
                [6.0, -1.1, 5, 'college'],
            ], ['stormy night']),
            new GradientBoost(),
            false,
        ];

        yield [
            Labeled::quick([
                [6.0, -1.1, 5, 'college'],
            ], [200]),
            new AdaBoost(),
            false,
        ];
    }
}
