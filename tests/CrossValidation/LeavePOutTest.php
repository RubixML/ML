<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\CrossValidation;

use Generator;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\CrossValidation\LeavePOut;
use Rubix\ML\Classifiers\GaussianNB;
use Rubix\ML\Datasets\Generators\Agglomerate;
use Rubix\ML\CrossValidation\Metrics\Accuracy;
use PHPUnit\Framework\TestCase;
use Rubix\ML\Backends\Backend;
use Rubix\ML\Backends\Amp;
use Rubix\ML\Backends\Swoole;
use Rubix\ML\Backends\Tasks\Task;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Specifications\ExtensionIsLoaded;
use Rubix\ML\Exceptions\InvalidArgumentException;
use PHPUnit\Framework\Attributes\RunInSeparateProcess;

#[Group('Validators')]
#[CoversClass(LeavePOut::class)]
class LeavePOutTest extends TestCase
{
    protected const int DATASET_SIZE = 50;

    protected Agglomerate $generator;

    protected GaussianNB $estimator;

    protected LeavePOut $validator;

    protected Accuracy $metric;

    protected ?Backend $backend = null;

    /**
     * @return Generator<string, array{backend: Backend}>
     */
    public static function provideBackends() : Generator
    {
        $ampBackend = new Amp();

        yield (string) $ampBackend => [
            'backend' => $ampBackend,
        ];

        if (ExtensionIsLoaded::with('swoole')->passes()) {
            $swooleBackend = new Swoole();

            yield (string) $swooleBackend => [
                'backend' => $swooleBackend,
            ];
        }
    }

    protected function setUp() : void
    {
        $this->generator = new Agglomerate(
            generators: [
                'male' => new Blob(
                    center: [69.2, 195.7, 40.0],
                    stdDev: [1.0, 3.0, 0.3]
                ),
                'female' => new Blob(
                    center: [63.7, 168.5, 38.1],
                    stdDev: [0.8, 2.5, 0.4]
                ),
            ],
            weights: [0.45, 0.55]
        );

        $this->estimator = new GaussianNB();

        $this->validator = new LeavePOut(10);

        $this->metric = new Accuracy();
    }

    protected function tearDown() : void
    {
        $this->backend?->shutdown();
    }

    #[DataProvider('provideBackends')]
    #[Test]
    #[RunInSeparateProcess]
    public function estimator(Backend $backend) : void
    {
        $this->backend = $backend;

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

    #[Test]
    public function tooFewSamplesThrows() : void
    {
        $this->expectException(InvalidArgumentException::class);

        $dataset = $this->generator->generate(5);

        $this->validator->test(
            estimator: $this->estimator,
            dataset: $dataset,
            metric: $this->metric
        );
    }

    #[Test]
    public function everySampleIsTestedAtLeastOnce() : void
    {
        foreach ([12 => 5, 10 => 3] as $n => $p) {
            $samples = $labels = [];

            for ($i = 0; $i < $n; ++$i) {
                // First feature column encodes the original sample index so the
                // tracking backend can recover which indices were held out.
                $samples[] = [(float) $i, (float) $i * 0.1, (float) $i * 0.2];
                $labels[] = $i % 2;
            }

            $dataset = Labeled::quick($samples, $labels);

            $backend = new TrackingBackend();
            $validator = new LeavePOut($p);
            $validator->setBackend($backend);

            $validator->test(
                estimator: new GaussianNB(),
                dataset: $dataset,
                metric: new Accuracy()
            );

            $tested = array_unique($backend->testedIndices);
            sort($tested);

            $this->assertSame(
                range(0, $n - 1),
                $tested,
                "Not all samples were tested for n=$n, p=$p."
            );
        }
    }
}

/**
 * A backend that records the original sample indices held out in each round.
 *
 * Samples carry their original index in the first feature column, so the backend
 * can recover which dataset positions were used as testing sets.
 */
class TrackingBackend implements Backend
{
    /**
     * The original sample indices that have been placed in a testing set.
     *
     * @var int[]
     */
    public array $testedIndices = [];

    /**
     * Dummy validation scores, one per enqueued round.
     *
     * @var list<float>
     */
    protected array $results = [];

    /**
     * @inheritDoc
     */
    public function enqueue(Task $task, ?callable $after = null) : void
    {
        $reflection = new \ReflectionProperty(Task::class, 'args');

        $args = $reflection->getValue($task);

        if (isset($args[2]) and $args[2] instanceof Labeled) {
            foreach ($args[2]->samples() as $sample) {
                $this->testedIndices[] = (int) $sample[0];
            }
        }

        $this->results[] = 0.5;
    }

    /**
     * @inheritDoc
     */
    public function workers() : int
    {
        return 1;
    }

    /**
     * @inheritDoc
     */
    public function process() : array
    {
        $results = $this->results;
        $this->results = [];

        return $results;
    }

    /**
     * @inheritDoc
     */
    public function flush() : void
    {
    }

    /**
     * @inheritDoc
     */
    public function shutdown() : void
    {
    }

    /**
     * @inheritDoc
     */
    public function __toString() : string
    {
        return 'TrackingBackend';
    }
}
