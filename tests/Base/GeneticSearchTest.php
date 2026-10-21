<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Base;

use Closure;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\TestCase;
use Rubix\ML\Classifiers\KNearestNeighbors;
use Rubix\ML\CrossValidation\Metrics\Accuracy;
use Rubix\ML\CrossValidation\Metrics\FBeta;
use Rubix\ML\CrossValidation\Metrics\Metric;
use Rubix\ML\CrossValidation\KFold;
use Rubix\ML\CrossValidation\Validator;
use Rubix\ML\Datasets\Generators\Agglomerate;
use Rubix\ML\Datasets\Generators\Circle;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Estimator;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use Rubix\ML\GeneticSearch;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\Learner;

/**
 * Genetic Search Tests
 */
#[Group('MetaEstimators')]
#[CoversClass(GeneticSearch::class)]
class GeneticSearchTest extends TestCase
{
    /**
     * Reset the random number generator before each test.
     */
    protected function setUp() : void
    {
        srand(42);
    }

    /**
     * Named parameters are reordered and omitted optional parameters use their defaults.
     */
    #[Test]
    public function fromNamedParamsFillsDefaults() : void
    {
        $search = GeneticSearch::fromNamedParams(
            KNearestNeighbors::class,
            [
                'kernel' => [new Euclidean()],
                'k' => [1, 3],
            ]
        );

        $this->assertEquals([
            'class' => KNearestNeighbors::class,
            'params' => [[1, 3], [false], [new Euclidean()]],
            'populationSize' => 16,
            'generations' => 10,
            'crossoverRate' => 0.8,
            'mutationRate' => 0.1,
            'eliteCount' => 1,
            'metric' => new FBeta(),
            'validator' => new KFold(5),
        ], $search->params());
    }

    /**
     * Empty parameter lists use the learner's constructor defaults.
     */
    #[Test]
    public function fillsEmptyTuplesWithDefaults() : void
    {
        $search = new GeneticSearch(KNearestNeighbors::class, [[], [], []]);

        $this->assertEquals([[5], [false], [null]], $search->params()['params']);
    }

    /**
     * An invalid population cannot produce two parents.
     */
    #[Test]
    public function rejectsInvalidPopulationSize() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new GeneticSearch(KNearestNeighbors::class, [[1]], populationSize: 1);
    }

    /**
     * Mutation probability must be a valid probability.
     */
    #[Test]
    public function rejectsInvalidMutationRate() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new GeneticSearch(KNearestNeighbors::class, [[1]], mutationRate: 1.1);
    }

    /**
     * Every supplied parameter must have a candidate list.
     */
    #[Test]
    public function rejectsScalarCandidates() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new GeneticSearch(KNearestNeighbors::class, [1]);
    }

    /**
     * Parameter values must be safe to persist with the search.
     */
    #[Test]
    public function rejectsUnserializableCandidates() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new GeneticSearch(KNearestNeighbors::class, [[Closure::fromCallable('strlen')]]);
    }

    /**
     * Values with different PHP types remain separate choices.
     */
    #[Test]
    public function preservesCandidateTypes() : void
    {
        $search = new GeneticSearch(KNearestNeighbors::class, [[1, '1', 1.0, true, 1]]);

        $this->assertSame([1, '1', 1.0, true], $search->params()['params'][0]);
    }

    /**
     * A resource cannot be restored from a serialized search.
     */
    #[Test]
    public function rejectsResourceCandidates() : void
    {
        $stream = fopen('php://memory', 'r+');

        $this->assertIsResource($stream);

        try {
            $this->expectException(InvalidArgumentException::class);

            new GeneticSearch(KNearestNeighbors::class, [[1, $stream]]);
        } finally {
            fclose($stream);
        }
    }

    /**
     * Results are only available after training.
     */
    #[Test]
    public function resultsThrowBeforeTraining() : void
    {
        $search = new GeneticSearch(KNearestNeighbors::class, [[1]]);

        $this->assertNull($search->scores());

        $this->expectException(RuntimeException::class);

        $search->results();
    }

    /**
     * Search evaluates each finite candidate once and selects the best fitness.
     */
    #[Test]
    public function exhaustsSmallSpaceWithoutRepeatedEvaluations() : void
    {
        $search = new GeneticSearch(
            class: KNearestNeighbors::class,
            params: [[1, 3, 5, 5], [false, true], [new Euclidean()]],
            populationSize: 4,
            generations: 5,
            metric: new FBeta(),
            validator: new GeneticSearchOracle()
        );

        $search->train($this->dataset());

        $scores = $search->scores();
        $rows = $search->results()->toArray();

        $this->assertNotNull($scores);
        $this->assertCount(6, $scores);
        $this->assertCount(6, $rows);
        $this->assertSame([1, 3, 5], $search->params()['params'][0]);

        $candidates = [];
        $metric = new FBeta();

        foreach ($scores as $i => $score) {
            $row = $rows['Trial ' . ($i + 1)];

            $this->assertSame((string) $score, $row["{$metric}"]);

            $candidate = serialize($row['params']);
            $this->assertArrayNotHasKey($candidate, $candidates);
            $candidates[$candidate] = true;
        }

        $this->assertSame(5, $search->base()->params()['k']);
        $this->assertTrue($search->base()->params()['weighted']);
        $this->assertTrue($search->trained());
    }

    /**
     * The selected learner is retrained and can make useful predictions.
     */
    #[Test]
    public function trainsAndPredicts() : void
    {
        $search = new GeneticSearch(
            class: KNearestNeighbors::class,
            params: [[1, 3, 5], [false, true], [new Euclidean()]],
            populationSize: 4,
            generations: 3
        );

        $training = $this->dataset(256);
        $testing = $this->dataset(128);

        $search->train($training);

        $predictions = $search->predict($testing);

        $this->assertCount($testing->numSamples(), $predictions);
        $this->assertGreaterThanOrEqual(
            0.85,
            (new Accuracy())->score($predictions, $testing->labels())
        );
    }

    /**
     * The setup hook runs on evaluated candidates and the final learner, but is transient.
     */
    #[Test]
    public function setupIsCalledAndExcludedFromSerialization() : void
    {
        $search = new GeneticSearch(
            class: KNearestNeighbors::class,
            params: [[1, 3], [false], [new Euclidean()]],
            populationSize: 4,
            generations: 3,
            validator: new GeneticSearchOracle()
        );

        $counter = new GeneticSearchSetupCounter();

        $search->setup(Closure::fromCallable([$counter, 'record']));

        $search->train($this->dataset());

        $scores = $search->scores();

        $this->assertNotNull($scores);
        $this->assertSame(count($scores) + 1, $counter->calls);
        $this->assertArrayNotHasKey('setup', $search->__serialize());

        $copy = unserialize(serialize($search));

        $this->assertInstanceOf(GeneticSearch::class, $copy);
        $this->assertTrue($copy->trained());
    }

    /**
     * Generate a labeled classification dataset for training tests.
     *
     * @param int $n
     * @return Labeled
     */
    protected function dataset(int $n = 128) : Labeled
    {
        $generator = new Agglomerate([
            'inner' => new Circle(x: 0.0, y: 0.0, scale: 1.0, noise: 0.5),
            'middle' => new Circle(x: 0.0, y: 0.0, scale: 5.0, noise: 1.0),
            'outer' => new Circle(x: 0.0, y: 0.0, scale: 10.0, noise: 2.0),
        ]);

        return $generator->generate($n);
    }
}

/**
 * Counts each learner configured by the setup hook.
 */
class GeneticSearchSetupCounter
{
    /**
     * Number of learners configured.
     */
    public int $calls = 0;

    /**
     * Count one configured learner.
     *
     * @param KNearestNeighbors $learner
     */
    public function record(KNearestNeighbors $learner) : void
    {
        ++$this->calls;
    }
}

/**
 * A deterministic fitness oracle for Genetic Search tests.
 */
class GeneticSearchOracle implements Validator
{
    /**
     * Score the selected KNN parameters with a unique optimum.
     *
     * @param Learner & Estimator $estimator
     * @param Labeled $dataset
     * @param Metric $metric
     * @return float
     */
    public function test(Learner & Estimator $estimator, Labeled $dataset, Metric $metric) : float
    {
        $params = $estimator->params();

        return ($params['k'] === 5 ? 0.5 : 0.0)
            + ($params['weighted'] ? 0.5 : 0.0);
    }

    /**
     * Return the name of the fitness oracle.
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Genetic Search Oracle';
    }
}
