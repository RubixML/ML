<?php

namespace Rubix\ML\Tests\Specifications;

use Rubix\ML\Estimator;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Classifiers\NaiveBayes;
use Rubix\ML\Regressors\RegressionTree;
use Rubix\ML\Clusterers\GaussianMixture;
use Rubix\ML\Specifications\SamplesAreCompatibleWithEstimator;
use PHPUnit\Framework\TestCase;
use InvalidArgumentException;
use Generator;

/**
 * @group Specifications
 * @covers \Rubix\ML\Specifications\SamplesAreCompatibleWithEstimator
 */
class SamplesAreCompatibleWithEstimatorTest extends TestCase
{
    /**
     * @test
     * @dataProvider checkProvider
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @param \Rubix\ML\Estimator $estimator
     * @param bool $valid
     */
    public function check(Dataset $dataset, Estimator $estimator, bool $valid) : void
    {
        if (!$valid) {
            $this->expectException(InvalidArgumentException::class);
        }

        SamplesAreCompatibleWithEstimator::check($dataset, $estimator);

        $this->assertTrue($valid);
    }

    /**
     * @return \Generator<array>
     */
    public function checkProvider() : Generator
    {
        yield [
            Unlabeled::quick([
                ['swamp', 'island', 'black knight', 'counter spell'],
            ]),
            new NaiveBayes(),
            true,
        ];

        yield [
            Unlabeled::quick([
                [6.0, -1.1, 5, 'college'],
            ]),
            new RegressionTree(),
            true,
        ];

        yield [
            Unlabeled::quick([
                [6.0, -1.1, 5, 'college'],
            ]),
            new NaiveBayes(),
            false,
        ];

        yield [
            Unlabeled::quick([
                [6.0, -1.1, 5, 'college'],
            ]),
            new GaussianMixture(3),
            false,
        ];
    }
}
