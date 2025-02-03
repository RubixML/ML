<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Specifications;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Classifiers\NaiveBayes;
use Rubix\ML\Regressors\RegressionTree;
use Rubix\ML\Clusterers\GaussianMixture;
use Rubix\ML\Specifications\SamplesAreCompatibleWithEstimator;
use PHPUnit\Framework\TestCase;
use Generator;

#[Group('Specifications')]
#[CoversClass(SamplesAreCompatibleWithEstimator::class)]
class SamplesAreCompatibleWithEstimatorTest extends TestCase
{
    public static function passesProvider() : Generator
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

    /**
     * @param SamplesAreCompatibleWithEstimator $specification
     * @param bool $expected
     */
    #[DataProvider('passesProvider')]
    public function testPasses(SamplesAreCompatibleWithEstimator $specification, bool $expected) : void
    {
        $this->assertSame($expected, $specification->passes());
    }
}
