<?php

namespace Rubix\ML\Clusterers;

use Rubix\ML\Learner;
use Rubix\ML\Verbose;
use Rubix\ML\DataType;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Probabilistic;
use Rubix\ML\EstimatorType;
use Rubix\ML\Helpers\Params;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Traits\LoggerAware;
use Rubix\ML\Clusterers\Seeders\KMC2;
use Rubix\ML\Traits\AutotrackRevisions;
use Rubix\ML\Kernels\Distance\Distance;
use Rubix\ML\Clusterers\Seeders\Seeder;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\Kernels\Distance\Symmetric;
use Rubix\ML\Specifications\DatasetIsNotEmpty;
use Rubix\ML\Specifications\SpecificationChain;
use Rubix\ML\Specifications\DatasetHasDimensionality;
use Rubix\ML\Specifications\SamplesAreCompatibleWithEstimator;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use Generator;

use function Rubix\ML\argmin;
use function count;
use function array_search;
use function array_fill;
use function get_object_vars;

use const Rubix\ML\EPSILON;

/**
 * K Medoids
 *
 * A robust medoid-based hard clustering algorithm capable of grouping linearly
 * separable data points given some prior knowledge of the target number of clusters
 * (defined by *k*). Unlike centroid-based algorithms such as [K Means](k-means.md),
 * K Medoids anchors each cluster with an *actual* sample of the training set (called
 * a *medoid*) rather than a mean vector, making the resultant clustering less
 * sensitive to outliers and noise.
 *
 * K Medoids follows the *CLARA* (*Clustering LARge Applications*) scheme: at each
 * epoch, a random subset of the training set is refined using the *PAM*
 * (*Partitioning Around Medoids*) heuristic to obtain a candidate set of medoids,
 * which is then scored against the inertia cost function on the **entire** dataset.
 * After *R* independent candidates have been proposed (controlled by the *epochs*
 * hyper-parameter), the candidate yielding the lowest full-dataset inertia is kept.
 *
 * This decouples the search space from the evaluation cost: PAM is run on the small
 * subset (O(n'²·k)) while the true objective — the full-dataset inertia — is what
 * ultimately selects the winning set of medoids.
 *
 * References:
 * [1] A. K. Jain et al. (1999). Data Clustering: A Review.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class KMedoids implements Estimator, Learner, Probabilistic, Verbose, Persistable
{
    use AutotrackRevisions, LoggerAware;

    /**
     * The target number of clusters.
     *
     * @var int<0,max>
     */
    protected int $k;

    /**
     * The size of the CLARA sample i.e. the number of samples drawn from the
     * training set to propose a candidate set of medoids at each epoch.
     *
     * @var positive-int
     */
    protected int $batchSize;

    /**
     * The number of CLARA iterations to run. Each iteration proposes an independent
     * candidate set of medoids; the best candidate (lowest full-dataset inertia)
     * is kept.
     *
     * @var int
     */
    protected int $epochs;

    /**
     * The minimum improvement in the total inertia required for a PAM SWAP exchange
     * to be accepted.
     *
     * @var float
     */
    protected float $minChange;

    /**
     * The distance kernel to use when computing the distances between samples.
     *
     * @var Distance
     */
    protected Distance $kernel;

    /**
     * The cluster medoid seeder.
     *
     * @var Seeder
     */
    protected Seeder $seeder;

    /**
     * The computed medoid vectors, i.e. the actual samples selected from the training
     * data.
     *
     * @var list<list<string|int|float>>
     */
    protected array $medoids = [
        //
    ];

    /**
     * The loss at each epoch from the last training session.
     *
     * @var float[]|null
     */
    protected ?array $losses = null;

    /**
     * @param int $k
     * @param int $batchSize
     * @param int $epochs
     * @param float $minChange
     * @param Distance|null $kernel
     * @param Seeder|null $seeder
     * @throws InvalidArgumentException
     */
    public function __construct(
        int $k,
        int $batchSize = 100,
        int $epochs = 10,
        float $minChange = 1e-4,
        ?Distance $kernel = null,
        ?Seeder $seeder = null
    ) {
        if ($k < 1) {
            throw new InvalidArgumentException('K must be greater'
                . " than 0, $k given.");
        }

        if ($batchSize < $k) {
            throw new InvalidArgumentException('Batch size must be greater'
                . " than or equal to $k, $batchSize given.");
        }

        if ($epochs < 1) {
            throw new InvalidArgumentException('Number of epochs'
                . " must be greater than 0, $epochs given.");
        }

        if ($minChange < 0.0) {
            throw new InvalidArgumentException('Minimum change must be'
                . " greater than 0, $minChange given.");
        }

        if (isset($kernel) and !$kernel instanceof Symmetric) {
            throw new InvalidArgumentException('Kernel must implement the Symmetric interface.');
        }

        $kernel ??= new Euclidean();

        $this->k = $k;
        $this->batchSize = $batchSize;
        $this->epochs = $epochs;
        $this->minChange = $minChange;
        $this->kernel = $kernel;
        $this->seeder = $seeder ?? new KMC2(kernel: $kernel);
    }

    /**
     * Return the estimator type.
     *
     * @return EstimatorType
     */
    public function type() : EstimatorType
    {
        return EstimatorType::clusterer();
    }

    /**
     * Return the data types that the estimator is compatible with.
     *
     * @return list<DataType>
     */
    public function compatibility() : array
    {
        return $this->kernel->compatibility();
    }

    /**
     * Return the settings of the hyper-parameters in an associative array.
     *
     * @return mixed[]
     */
    public function params() : array
    {
        return [
            'k' => $this->k,
            'batch size' => $this->batchSize,
            'epochs' => $this->epochs,
            'min change' => $this->minChange,
            'kernel' => $this->kernel,
            'seeder' => $this->seeder,
        ];
    }

    /**
     * Has the learner been trained?
     *
     * @return bool
     */
    public function trained() : bool
    {
        return !empty($this->medoids);
    }

    /**
     * Return the computed cluster medoids, i.e. the actual samples of the training
     * set that anchor each cluster.
     *
     * @return list<list<string|int|float>>
     */
    public function medoids() : array
    {
        return $this->medoids;
    }

    /**
     * Return an iterable progress table with the steps from the last training session.
     *
     * @return Generator<mixed[]>
     */
    public function steps() : Generator
    {
        if (!$this->losses) {
            return;
        }

        foreach ($this->losses as $epoch => $loss) {
            yield [
                'epoch' => $epoch,
                'loss' => $loss,
            ];
        }
    }

    /**
     * Return the loss for each epoch from the last training session.
     *
     * @return float[]|null
     */
    public function losses() : ?array
    {
        return $this->losses;
    }

    /**
     * Train the learner with a dataset.
     *
     * The CLARA loop runs *R* (i.e. *epochs*) independent iterations. Each
     * iteration proposes a candidate set of medoids by seeding a random subset
     * and refining it with PAM to convergence, then evaluates the candidate on the
     * **entire** dataset. The candidate yielding the lowest full-dataset inertia
     * is kept.
     *
     * @param Dataset $dataset
     * @throws InvalidArgumentException
     */
    public function train(Dataset $dataset) : void
    {
        SpecificationChain::with([
            new DatasetIsNotEmpty($dataset),
            new SamplesAreCompatibleWithEstimator($dataset, $this),
        ])->check();

        if ($this->logger) {
            $this->logger->info("Training $this");
        }

        $numSamples = $dataset->numSamples();

        if ($numSamples < $this->k) {
            throw new InvalidArgumentException("Dataset must contain at least {$this->k}"
                . " samples, $numSamples given.");
        }

        $this->losses = [];

        $subsetSize = min($this->batchSize, $numSamples);

        $bestMedoids = null;
        $bestLoss = INF;

        for ($epoch = 1; $epoch <= $this->epochs; ++$epoch) {
            $subset = $dataset->randomSubset($subsetSize);

            $seeds = $this->seeder->seed($subset, $this->k);

            $medoids = [];

            foreach ($seeds as $seed) {
                $offset = array_search($seed, $subset->samples());

                $medoids[] = is_int($offset) ? $offset : 0;
            }

            $distances = $this->distanceMatrix($subset);

            $loss = $this->totalInertia($medoids, $distances);

            do {
                $improved = false;

                for ($i = 0; $i < count($medoids); ++$i) {
                    $bestDelta = -$this->minChange;
                    $bestOffset = null;

                    $candidates = $medoids;

                    for ($j = 0; $j < $subset->numSamples(); ++$j) {
                        if (in_array($j, $medoids)) {
                            continue;
                        }

                        $candidates[$i] = $j;

                        $delta = $this->totalInertia($candidates, $distances) - $loss;

                        if ($delta < $bestDelta) {
                            $bestDelta = $delta;

                            $bestOffset = $j;
                        }
                    }

                    if (isset($bestOffset)) {
                        $medoids[$i] = $bestOffset;

                        $loss += $bestDelta;

                        $improved = true;
                    }
                }
            } while ($improved);

            $candidates = array_map(fn ($offset) => $subset->samples()[$offset], $medoids);

            $sum = 0.0;

            foreach ($dataset->samples() as $sample) {
                $min = INF;

                foreach ($candidates as $medoid) {
                    $distance = $this->kernel->compute($sample, $medoid);

                    if ($distance < $min) {
                        $min = $distance;
                    }
                }

                $sum += $min;
            }

            $loss = $sum / $dataset->numSamples();

            $this->losses[$epoch] = $loss;

            if ($this->logger) {
                $message = "Epoch: $epoch, Inertia: $loss";

                $this->logger->info($message);
            }

            if (is_nan($loss)) {
                if ($this->logger) {
                    $this->logger->warning('Numerical instability detected');
                }

                break;
            }

            if ($loss < $bestLoss) {
                $bestLoss = $loss;

                $bestMedoids = $candidates;
            }
        }

        if ($bestMedoids === null) {
            throw new RuntimeException('No candidates were proposed during training.');
        }

        $this->medoids = $bestMedoids;

        if ($this->logger) {
            $this->logger->info('Training complete');
        }
    }

    /**
     * Cluster the dataset by assigning a label to each sample.
     *
     * @param Dataset $dataset
     * @throws RuntimeException
     * @return list<int>
     */
    public function predict(Dataset $dataset) : array
    {
        if (!$this->medoids) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        DatasetHasDimensionality::with($dataset, count(current($this->medoids)))->check();

        return array_map([$this, 'predictSample'], $dataset->samples());
    }

    /**
     * Label a given sample based on its distance from each cluster medoid.
     *
     * @internal
     *
     * @param list<string|int|float> $sample
     * @return int
     */
    public function predictSample(array $sample) : int
    {
        return argmin($this->medoidDistances($sample));
    }

    /**
     * Estimate the joint probabilities for each possible outcome.
     *
     * @param Dataset $dataset
     * @throws RuntimeException
     * @return list<float[]>
     */
    public function proba(Dataset $dataset) : array
    {
        if (!$this->medoids) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        DatasetHasDimensionality::with($dataset, count(current($this->medoids)))->check();

        return array_map([$this, 'probaSample'], $dataset->samples());
    }

    /**
     * Return the membership of a sample to each of the k cluster medoids.
     *
     * @internal
     *
     * @param list<string|int|float> $sample
     * @return float[]
     */
    public function probaSample(array $sample) : array
    {
        $distances = $proba = [];

        foreach ($this->medoids as $medoid) {
            $distances[] = $this->kernel->compute($sample, $medoid) ?: EPSILON;
        }

        foreach ($distances as $distanceA) {
            $sigma = 0.0;

            foreach ($distances as $distanceB) {
                $sigma += $distanceA / $distanceB;
            }

            $proba[] = 1.0 / $sigma;
        }

        return $proba;
    }

    /**
     * Compute the distance from a sample to each cluster medoid.
     *
     * @internal
     *
     * @param list<string|int|float> $sample
     * @return list<float>
     */
    protected function medoidDistances(array $sample) : array
    {
        $distances = [];

        foreach ($this->medoids as $medoid) {
            $distances[] = $this->kernel->compute($sample, $medoid);
        }

        return $distances;
    }

    /**
     * Compute the total inertia of the sample assignments given a set of medoids.
     *
     * @param list<int> $medoids
     * @param list<list<float>> $distances
     * @return float
     */
    protected function totalInertia(array $medoids, array $distances) : float
    {
        $sum = 0.0;

        foreach ($distances as $row) {
            $min = INF;

            foreach ($medoids as $offset) {
                if ($row[$offset] < $min) {
                    $min = $row[$offset];
                }
            }

            $sum += $min;
        }

        return $sum;
    }

    /**
     * Compute the distance matrix of the samples i.e. the pairwise distance between
     * every pair of samples in the data set.
     *
     * @param Dataset $dataset
     * @return list<list<float>>
     */
    protected function distanceMatrix(Dataset $dataset) : array
    {
        $n = $dataset->numSamples();

        $matrix = array_fill(0, $n, []);

        for ($i = 0; $i < $n; ++$i) {
            for ($j = $i + 1; $j < $n; ++$j) {
                $distance = $this->kernel->compute($dataset->sample($i), $dataset->sample($j)) ?: EPSILON;

                $matrix[$i][$j] = $distance;
                $matrix[$j][$i] = $distance;
            }

            $matrix[$i][$i] = 0.0;
        }

        return $matrix;
    }

    /**
     * Return an associative array containing the data used to serialize the object.
     *
     * @return mixed[]
     */
    public function __serialize() : array
    {
        $properties = get_object_vars($this);

        unset($properties['losses'], $properties['logger']);

        return $properties;
    }

    /**
     * Restore the object from an associative array of serialized properties.
     *
     * @param mixed[] $properties
     */
    public function __unserialize(array $properties) : void
    {
        foreach ($properties as $property => $value) {
            $this->{$property} = $value;
        }
    }

    /**
     * Return the string representation of the object.
     *
     * @internal
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'K Medoids (' . Params::stringify($this->params()) . ')';
    }
}
