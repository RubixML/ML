<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\CrossValidation;

use Generator;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\CrossValidation\KFold;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Classifiers\GaussianNB;
use Rubix\ML\Datasets\Generators\Agglomerate;
use Rubix\ML\CrossValidation\Metrics\Accuracy;
use PHPUnit\Framework\TestCase;
use Rubix\ML\Backends\Backend;
use Rubix\ML\Backends\Serial;
use Rubix\ML\Backends\Swoole;
use Rubix\ML\Specifications\ExtensionIsLoaded;
use Rubix\ML\Specifications\SwooleExtensionIsLoaded;

#[Group('Validators')]
#[CoversClass(KFold::class)]
class KFoldTest extends TestCase
{
    protected const int DATASET_SIZE = 50;

    protected Agglomerate $generator;

    protected GaussianNB $estimator;

    protected KFold $validator;

    protected Accuracy $metric;

    /**
     * @return Generator<string, array{backend: Backend}>
     */
    public static function provideBackends() : Generator
    {
        $serialBackend = new Serial();

        yield (string) $serialBackend => [
            'backend' => $serialBackend,
        ];

        if (
            SwooleExtensionIsLoaded::create()->passes()
            && ExtensionIsLoaded::with('igbinary')->passes()
        ) {
            $swooleProcessBackend = new Swoole();

            yield (string) $swooleProcessBackend => [
                'backend' => $swooleProcessBackend,
            ];
        }
    }

    protected function setUp() : void
    {
        $this->generator = new Agglomerate(
            generators: [
                'male' => new Blob(
                    center: [69.2, 195.7, 40.],
                    stdDev: [1., 3., 0.3]
                ),
                'female' => new Blob(
                    center: [63.7, 168.5, 38.1],
                    stdDev: [0.8, 2.5, 0.4]
                ),
            ],
            weights: [0.45, 0.55]
        );

        $this->estimator = new GaussianNB();

        $this->validator = new KFold(10);

        $this->metric = new Accuracy();
    }

    #[DataProvider('provideBackends')]
    public function testTestEstimator(Backend $backend) : void
    {
        $this->validator->setBackend($backend);

        [$min, $max] = $this->metric->range()->list();

        $dataset = $this->generator->generate(self::DATASET_SIZE);

        $score = $this->validator->test(
            estimator: $this->estimator,
            dataset: $dataset,
            metric: $this->metric
        );

        $this->assertThat(
            $score,
            $this->logicalAnd(
                $this->greaterThanOrEqual($min),
                $this->lessThanOrEqual($max)
            )
        );
    }
}
