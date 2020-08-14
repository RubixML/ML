<?php

namespace Rubix\ML\Tests\Specifications;

use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Classifiers\NaiveBayes;
use Rubix\ML\Regressors\RegressionTree;
use Rubix\ML\Clusterers\GaussianMixture;
use Rubix\ML\Specifications\Specification;
use Rubix\ML\Specifications\SamplesAreCompatibleWithEstimator;
use PHPUnit\Framework\TestCase;
use Generator;

/**
 * @group Specifications
 * @covers \Rubix\ML\Specifications\SamplesAreCompatibleWithEstimator
 */
class SamplesAreCompatibleWithEstimatorTest extends TestCase
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
            SamplesAreCompatibleWithEstimator::with(
                Unlabeled::quick([
                    ['swamp', 'island', 'black knight', 'counter spell'],
                ]),
                new NaiveBayes()
            ),
            true,
        ];

        yield [
            SamplesAreCompatibleWithEstimator::with(
                Unlabeled::quick([
                    [6.0, -1.1, 5, 'college'],
                ]),
                new RegressionTree()
            ),
            true,
        ];

        yield [
            SamplesAreCompatibleWithEstimator::with(
                Unlabeled::quick([
                    [6.0, -1.1, 5, 'college'],
                ]),
                new NaiveBayes()
            ),
            false,
        ];

        yield [
            SamplesAreCompatibleWithEstimator::with(
                Unlabeled::quick([
                    [6.0, -1.1, 5, 'college'],
                ]),
                new GaussianMixture(3)
            ),
            false,
        ];
    }
}
