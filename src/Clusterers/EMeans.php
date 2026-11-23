<?php

namespace Rubix\ML\Clusterers;

use Rubix\ML\DataType;
use Rubix\ML\Learner;
use Rubix\ML\Verbose;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Probabilistic;
use Rubix\ML\EstimatorType;
use Rubix\ML\Helpers\Params;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Traits\LoggerAware;
use Rubix\ML\Traits\AutotrackRevisions;
use Rubix\ML\Kernels\Distance\Distance;
use Rubix\ML\Clusterers\Seeders\Seeder;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\Kernels\Distance\SafeEuclidean;
use Rubix\ML\Clusterers\Seeders\Random;
use Rubix\ML\Specifications\DatasetIsNotEmpty;
use Rubix\ML\Specifications\ExtensionIsLoaded;
use Rubix\ML\Specifications\SpecificationChain;
use Rubix\ML\Specifications\ExtensionMinimumVersion;
use Rubix\ML\Specifications\SamplesAreCompatibleWithEstimator;
use Rubix\ML\Specifications\DatasetHasDimensionality;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\MissingExtension;
use Rubix\ML\Exceptions\RuntimeException;
use Tensor\ColumnVector;
use Tensor\Matrix;
use Tensor\Vector;
use Generator;

use function Rubix\ML\argmin;
use function count;
use function floor;
use function is_finite;
use function array_fill;
use function array_map;
use function array_keys;
use function arsort;
use function get_object_vars;
use function in_array;
use function implode;
use function intdiv;
use function log;
use function min;
use function max;
use function pow;
use function round;
use function sqrt;
use function cos;

use const M_PI;
use const Rubix\ML\EPSILON;

/**
 * E Means
 *
 * A mini-batch centroid clustering algorithm in the spirit of E-Means, the
 * scaling-law analysis of mini-batch K-Means. The statistical batch size is
 * derived from the dataset geometry via the fitted batch laws, the initial
 * exponential-moving-average (EMA) step is derived via the fitted step law,
 * and the EMA step is annealed along a cosine schedule over the run. Each
 * mini batch updates the per-cluster sums and masses of the exponential
 * moving average, clusters that hold less than a fraction of the mean mass
 * are re-seeded from the farthest members of the heaviest cluster, and the
 * running centroids are re-derived as the exact means of the members of the
 * closing pass.
 *
 * The per-batch linear algebra is performed by the Tensor C extension, which
 * must be loaded at version 4.1.0 or higher for this estimator to function.
 *
 * References:
 * [1] E-Means: Scaling Laws for Mini-Batch K-Means.
 *     https://arxiv.org/abs/2611.09501
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class EMeans implements Estimator, Learner, Probabilistic, Verbose, Persistable
{
    use AutotrackRevisions, LoggerAware;

    /**
     * The list of allowed distance kernels for the E Means algorithm.
     */
    public const array ALLOWED_KERNELS = [
        Euclidean::class,
        SafeEuclidean::class,
    ];

    /**
     * The initial mass assigned to every cluster and the numerical floor
     * that keeps cluster mass strictly positive throughout the run.
     *
     * @var float
     */
    protected const float INITIAL_MASS = 1.0e-10;

    /**
     * Dimensions and below use the quality batch law and above it the
     * throughput law.
     *
     * @var int
     */
    protected const int QUALITY_MAX_DIM = 16;

    /**
     * The coefficient of the quality batch law.
     *
     * @var float
     */
    protected const float QUALITY_BATCH_COEFFICIENT = 0.00439175;

    /**
     * The coefficient of the throughput batch law.
     *
     * @var float
     */
    protected const float THROUGHPUT_BATCH_COEFFICIENT = 0.035134;

    /**
     * The k exponent of the batch laws.
     *
     * @var float
     */
    protected const float BATCH_LAW_K = 0.134749;

    /**
     * The n exponent of the batch laws.
     *
     * @var float
     */
    protected const float BATCH_LAW_N = 0.876473;

    /**
     * The T exponent of the batch laws.
     *
     * @var float
     */
    protected const float BATCH_LAW_T = 0.813344;

    /**
     * The coefficient of the step law.
     *
     * @var float
     */
    protected const float STEP_LAW_COEFFICIENT = 0.8244;

    /**
     * The k exponent of the step law.
     *
     * @var float
     */
    protected const float STEP_LAW_K = -0.4617;

    /**
     * The B exponent of the step law.
     *
     * @var float
     */
    protected const float STEP_LAW_B = 0.9003;

    /**
     * The n exponent of the step law.
     *
     * @var float
     */
    protected const float STEP_LAW_N = -0.4498;

    /**
     * The T exponent of the step law.
     *
     * @var float
     */
    protected const float STEP_LAW_T = -0.1057;

    /**
     * The lower bound of the EMA step given by the step law.
     *
     * @var float
     */
    protected const float STEP_LAW_MIN = 1.0e-4;

    /**
     * The fraction of the initial step remaining at the end of the cosine
     * schedule.
     *
     * @var float
     */
    protected const float SCHEDULE_FLOOR = 0.1;

    /**
     * The number of logical updates between revivals.
     *
     * @var int
     */
    protected const int REVIVAL_INTERVAL = 5;

    /**
     * A cluster is considered dead when its mass falls below this fraction
     * of the mean mass.
     *
     * @var float
     */
    protected const float DEAD_MASS_THRESHOLD = 0.1;

    /**
     * The target number of clusters.
     *
     * @var int<0,max>
     */
    protected int $k;

    /**
     * The target number of training passes over the dataset; a value below
     * one restricts the EMA phase to a fractional prefix of the dataset.
     *
     * @var float
     */
    protected float $epochs;

    /**
     * The size of each mini batch in samples, or null to derive it from
     * the dataset geometry with the batch laws.
     *
     * @var positive-int|null
     */
    protected ?int $batchSize = null;

    /**
     * The initial EMA step, or null to derive it from the dataset geometry
     * with the step law.
     *
     * @var positive-float|null
     */
    protected ?float $alpha0 = null;

    /**
     * The distance function to use when computing the distances.
     *
     * @var Distance
     */
    protected Distance $kernel;

    /**
     * The cluster centroid seeder.
     *
     * @var Seeder
     */
    protected Seeder $seeder;

    /**
     * The computed centroid vectors of the training data.
     *
     * @var list<list<int|float>>
     */
    protected array $centroids = [
        //
    ];

    /**
     * The number of training samples contained within each cluster centroid.
     *
     * @var int[]
     */
    protected array $sizes = [
        //
    ];

    /**
     * The batch size resolved for the last training session.
     *
     * @var positive-int|null
     */
    protected ?int $resolvedBatch = null;

    /**
     * The initial EMA step resolved for the last training session.
     *
     * @var positive-float|null
     */
    protected ?float $resolvedAlpha0 = null;

    /**
     * The loss at each epoch from the last training session.
     *
     * @var float[]|null
     */
    protected ?array $losses = null;

    /**
     * The running sums of the feature values per cluster (c = S / N).
     *
     * @var Matrix|null
     */
    protected ?Matrix $sums = null;

    /**
     * The running EMA mass per cluster.
     *
     * @var ColumnVector|null
     */
    protected ?ColumnVector $masses = null;

    /**
     * Round a positive value to the nearest power of two.
     *
     * @internal
     *
     * @param float $value
     * @return positive-int
     */
    protected static function roundPowerOfTwo(float $value) : int
    {
        if ($value <= 1.0) {
            return 1;
        }

        $low = 1 << (int) floor(log($value, 2));

        $high = 2 * $low;

        return $value / $low < $high / $value ? $low : $high;
    }

    /**
     * Build an E Means estimator.
     *
     * @param int $k
     * @param float $epochs
     * @param int|null $batchSize
     * @param float|null $alpha0
     * @param Distance|null $kernel
     * @param Seeder|null $seeder
     * @throws InvalidArgumentException
     */
    public function __construct(
        int $k,
        float $epochs = 5.0,
        ?int $batchSize = null,
        ?float $alpha0 = null,
        ?Distance $kernel = null,
        ?Seeder $seeder = null
    ) {
        if ($k < 1) {
            throw new InvalidArgumentException('K must be greater'
                . " than 0, $k given.");
        }

        if (!is_finite($epochs)) {
            throw new InvalidArgumentException('Number of epochs must be'
                . ' finite.');
        }

        if ($epochs <= 0.0) {
            throw new InvalidArgumentException('Number of epochs must be'
                . " greater than 0, $epochs given.");
        }

        if (isset($batchSize) and $batchSize < 1) {
            throw new InvalidArgumentException('Batch size must be'
                . " greater than 0, $batchSize given.");
        }

        if (isset($alpha0) and !is_finite($alpha0)) {
            throw new InvalidArgumentException('Initial step must be finite.');
        }

        if (isset($alpha0) and $alpha0 <= 0.0) {
            throw new InvalidArgumentException('Initial step must be a'
                . " positive number, $alpha0 given.");
        }

        if (isset($kernel) and !in_array($kernel::class, self::ALLOWED_KERNELS)) {
            throw new InvalidArgumentException('Kernel must be one of: '
                . implode(', ', self::ALLOWED_KERNELS) . ", $kernel given.");
        }

        $this->k = $k;
        $this->epochs = $epochs;
        $this->batchSize = $batchSize;
        $this->alpha0 = $alpha0;
        $this->kernel = $kernel ?? new Euclidean();
        $this->seeder = $seeder ?? new Random();
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
        return [
            DataType::continuous(),
        ];
    }

    /**
     * Return the settings of the hyper-parameters in an associative array.
     *
     * @return mixed[]
     */
    public function params() : array
    {
        $params = [
            'k' => $this->k,
            'epochs' => $this->epochs,
            'batch size' => $this->batchSize,
            'initial step' => $this->alpha0,
            'kernel' => $this->kernel,
            'seeder' => $this->seeder,
        ];

        if ($this->trained()) {
            $params['resolved batch size'] = $this->resolvedBatch;

            $params['resolved initial step'] = $this->resolvedAlpha0;
        }

        return $params;
    }

    /**
     * Has the learner been trained?
     *
     * @return bool
     */
    public function trained() : bool
    {
        return !empty($this->centroids);
    }

    /**
     * Return the computed cluster centroids of the training data.
     *
     * @return list<list<int|float>>
     */
    public function centroids() : array
    {
        return $this->centroids;
    }

    /**
     * Return the number of training samples contained within each cluster centroid.
     *
     * @return int[]
     */
    public function sizes() : array
    {
        return $this->sizes;
    }

    /**
     * Return an iterable progress table from the last training session.
     *
     * @return Generator<mixed[]>
     */
    public function progress() : Generator
    {
        if (!$this->losses) {
            return;
        }

        foreach ($this->losses as $epoch => $loss) {
            yield [
                'Epoch' => $epoch,
                'Inertia' => $loss,
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
     * @param Dataset $dataset
     * @throws MissingExtension
     */
    public function train(Dataset $dataset) : void
    {
        SpecificationChain::with([
            new DatasetIsNotEmpty($dataset),
            new SamplesAreCompatibleWithEstimator($dataset, $this),
            ExtensionIsLoaded::with('tensor'),
            ExtensionMinimumVersion::with('tensor', '4.1.0'),
        ])->check();

        if ($this->logger) {
            $this->logger->info("Training $this");
        }

        $n = $dataset->numSamples();

        $d = $dataset->numFeatures();

        $batch = $this->resolveBatchSize($n, $d);

        $alpha0 = $this->resolveAlpha0($n, $batch);

        $rowCounts = $this->epochRowCounts($n);

        $totalSteps = 0;

        foreach ($rowCounts as $rows) {
            $totalSteps += intdiv($rows + $batch - 1, $batch);
        }

        /** @var list<list<string|int|float>> $seeds */
        $seeds = $this->seeder->seed($dataset, $this->k);

        $this->sums = Matrix::fromArray($seeds, false)->multiplyScalar(self::INITIAL_MASS);

        $this->masses = ColumnVector::fromArray(
            array_fill(0, $this->k, self::INITIAL_MASS),
            false
        );

        $this->resolvedBatch = $batch;

        $this->resolvedAlpha0 = $alpha0;

        $this->losses = [];

        $step = 0;

        foreach ($rowCounts as $epoch => $rows) {
            $source = $rows < $n ? $dataset->randomSubset($rows) : $dataset;

            $epochLoss = 0.0;

            $epochRows = 0;

            foreach ($source->randomize()->batch($batch) as $miniBatch) {
                /** @var list<list<string|int|float>> $samples */
                $samples = $miniBatch->samples();

                $size = count($samples);

                [$loss, $labels] = $this->step($samples, $this->scheduledAlpha($alpha0, $step, $totalSteps));

                $epochLoss += $loss * $size;

                $epochRows += $size;

                ++$step;

                if ($step % self::REVIVAL_INTERVAL === 0) {
                    $this->reviveDeadClusters($samples, $labels);
                }
            }

            $loss = $epochLoss / $epochRows;

            $this->losses[$epoch + 1] = $loss;

            if ($this->logger) {
                $message = 'Epoch: ' . ($epoch + 1) . ", Inertia: $loss";

                $this->logger->info($message);
            }
        }

        $this->finalize($dataset, $n, $rowCounts);

        $this->sums = null;

        $this->masses = null;

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
        if (!$this->centroids) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        DatasetHasDimensionality::with($dataset, count(current($this->centroids)))->check();

        return array_map([$this, 'predictSample'], $dataset->samples());
    }

    /**
     * Label a given sample based on its distance from a particular centroid.
     *
     * @internal
     *
     * @param list<int|float> $sample
     * @return int
     */
    public function predictSample(array $sample) : int
    {
        return argmin($this->centroidDistances($sample));
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
        if (!$this->centroids) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        DatasetHasDimensionality::with($dataset, count(current($this->centroids)))->check();

        return array_map([$this, 'probaSample'], $dataset->samples());
    }

    /**
     * Return the membership of a sample to each of the k centroids.
     *
     * @internal
     *
     * @param list<int|float> $sample
     * @return float[]
     */
    public function probaSample(array $sample) : array
    {
        $distances = $dist = [];

        foreach ($this->centroids as $centroid) {
            $distances[] = $this->kernel->compute($sample, $centroid) ?: EPSILON;
        }

        foreach ($distances as $distanceA) {
            $sigma = 0.0;

            foreach ($distances as $distanceB) {
                $sigma += $distanceA / $distanceB;
            }

            $dist[] = 1.0 / $sigma;
        }

        return $dist;
    }

    /**
     * Derive the statistical batch size B from the batch laws at the given
     * dataset size and feature dimensionality.
     *
     * @internal
     *
     * @param positive-int $n
     * @param positive-int $d
     * @return positive-int
     */
    protected function resolveBatchSize(int $n, int $d) : int
    {
        if (null !== $this->batchSize) {
            return min($this->batchSize, $n);
        }

        if ($d <= self::QUALITY_MAX_DIM) {
            $coefficient = self::QUALITY_BATCH_COEFFICIENT;

            $cap = intdiv($n, 2);
        } else {
            $coefficient = self::THROUGHPUT_BATCH_COEFFICIENT;

            $cap = self::roundPowerOfTwo($n / 2);
        }

        $raw = $coefficient
            * pow($this->k, self::BATCH_LAW_K)
            * pow($n, self::BATCH_LAW_N)
            * pow($this->epochs, self::BATCH_LAW_T);

        $batch = self::roundPowerOfTwo($raw);

        return max(1, min($batch, $cap, $n));
    }

    /**
     * Derive the initial EMA step from the step law at the resolved batch
     * size, clamped to a strictly positive range.
     *
     * @internal
     *
     * @param positive-int $n
     * @param positive-int $batch
     * @return positive-float
     */
    protected function resolveAlpha0(int $n, int $batch) : float
    {
        if (null !== $this->alpha0) {
            return min($this->alpha0, 1.0 - self::INITIAL_MASS);
        }

        $alpha0 = self::STEP_LAW_COEFFICIENT
            * pow($this->k, self::STEP_LAW_K)
            * pow($batch, self::STEP_LAW_B)
            * pow($n, self::STEP_LAW_N)
            * pow($this->epochs, self::STEP_LAW_T);

        return min(max($alpha0, self::STEP_LAW_MIN), 1.0 - self::INITIAL_MASS);
    }

    /**
     * Derive the row counts of each pass: the whole passes, a rounded
     * fractional prefix, and the closing pass.
     *
     * @internal
     *
     * @param positive-int $n
     * @return list<positive-int>
     */
    protected function epochRowCounts(int $n) : array
    {
        $whole = (int) floor($this->epochs);

        $remainder = (int) round(($this->epochs - $whole) * $n);

        if ($whole < 1) {
            return [max(1, $remainder)];
        }

        $rowCounts = array_fill(0, $whole - 1, $n);

        if ($remainder) {
            $rowCounts[] = $remainder;
        }

        $rowCounts[] = $n;

        return $rowCounts;
    }

    /**
     * Cosine-anneal the initial EMA step towards the schedule floor.
     *
     * @internal
     *
     * @param positive-float $alpha0
     * @param non-negative-int $step
     * @param non-negative-int $total
     * @return positive-float
     */
    protected function scheduledAlpha(float $alpha0, int $step, int $total) : float
    {
        $fraction = min($step, $total) / max(1, $total);

        $cosine = 0.5 * (1.0 + cos(M_PI * $fraction));

        return $alpha0 * (self::SCHEDULE_FLOOR + (1.0 - self::SCHEDULE_FLOOR) * $cosine);
    }

    /**
     * Assign a mini batch to the running centroids, fold its additive
     * statistics into the exponential moving average, and return the batch
     * inertia alongside its assignment labels.
     *
     * @internal
     *
     * @param list<list<string|int|float>> $samples
     * @param positive-float $alpha
     * @return array{float, list<int>}
     */
    protected function step(array $samples, float $alpha) : array
    {
        [$labels, $s, $n, $x2, $d2] = $this->batchStats($samples);

        $decay = 1.0 - $alpha;

        $this->sums = $this->sums->multiplyScalar($decay)->addMatrix($s->multiplyScalar($alpha));

        $this->masses = $this->masses->multiplyScalar($decay)->add($n->multiplyScalar($alpha));

        $inertia = 0.0;

        $x2Rows = $x2->asArray();

        $minD2 = $d2->min()->asArray();

        foreach ($x2Rows as $i => $x) {
            $inertia += sqrt($x + $minD2[$i]);
        }

        return [$inertia / count($samples), $labels];
    }

    /**
     * Assign the samples to the running centroids and collect the additive
     * batch statistics used by the moving average: the labels, the
     * per-cluster sums, the per-cluster counts, the per-sample squared
     * norms, and the per-sample squared distances to the running centroids.
     *
     * @internal
     *
     * @param list<list<string|int|float>> $samples
     * @return array{list<int>, Matrix, ColumnVector, ColumnVector, Matrix}
     */
    protected function batchStats(array $samples) : array
    {
        $size = count($samples);

        $X = Matrix::fromArray($samples, false);

        $d2 = $this->squaredDistances($samples);

        $x2 = $X->square()->sum();

        $labels = $this->rowArgmins($d2);

        $M = ColumnVector::fromArray(array_map('floatval', $labels), false)
            ->outer(Vector::ones($this->k))
            ->equal(Vector::range(0, $this->k - 1)->outer(Vector::ones($size))->transpose());

        return [
            $labels,
            $M->transpose()->matmul($X),
            $M->transpose()->sum(),
            $x2,
            $d2,
        ];
    }

    /**
     * Compute the index of the minimum value in each row of the given
     * matrix.
     *
     * @internal
     *
     * @param Matrix $matrix
     * @return list<int>
     */
    protected function rowArgmins(Matrix $matrix) : array
    {
        $labels = [];

        foreach ($matrix->asArray() as $row) {
            $labels[] = argmin($row);
        }

        return $labels;
    }

    /**
     * Compute the squared distance from each sample to each running
     * centroid.
     *
     * @internal
     *
     * @param list<list<string|int|float>> $samples
     * @return Matrix
     */
    protected function squaredDistances(array $samples) : Matrix
    {
        $X = Matrix::fromArray($samples, false);

        $C = $this->sums->divideColumnVector($this->masses);

        return $X->matmul($C->transpose())
            ->multiplyScalar(-2.0)
            ->addVector(Vector::fromArray($C->square()->sum()->asArray(), false))
            ->addColumnVector($X->square()->sum());
    }

    /**
     * Assign the samples to the running centroids.
     *
     * @internal
     *
     * @param list<list<string|int|float>> $samples
     * @return list<int>
     */
    protected function assignSamples(array $samples) : array
    {
        return $this->rowArgmins($this->squaredDistances($samples));
    }

    /**
     * Re-seed the dead clusters from the farthest members of the heaviest
     * cluster in the most recent mini batch, keeping their running mass.
     *
     * @internal
     *
     * @param list<list<string|int|float>> $samples
     * @param list<int> $labels
     */
    protected function reviveDeadClusters(array $samples, array $labels) : void
    {
        $massArr = $this->masses->asArray();

        $total = 0.0;

        foreach ($massArr as $mass) {
            $total += $mass;
        }

        if ($total <= 0.0) {
            return;
        }

        $tau = self::DEAD_MASS_THRESHOLD * $total / $this->k;

        $dead = [];

        foreach ($massArr as $j => $mass) {
            if ($mass < $tau) {
                $dead[] = $j;
            }
        }

        if (!$dead) {
            return;
        }

        $donor = 0;

        $peak = $massArr[0];

        foreach ($massArr as $j => $mass) {
            if ($mass > $peak) {
                $peak = $mass;
                $donor = $j;
            }
        }

        $C = $this->sums->divideColumnVector($this->masses);

        $donorC = $C->rowAsVector($donor)->asArray();

        $farthest = [];

        foreach ($labels as $i => $label) {
            if ($label !== $donor) {
                continue;
            }

            $row = $samples[$i];

            $dist2 = 0.0;

            foreach ($row as $f => $x) {
                $delta = $x - $donorC[$f];

                $dist2 += $delta * $delta;
            }

            $farthest[$i] = $dist2;
        }

        if (!$farthest) {
            return;
        }

        arsort($farthest);

        $order = array_keys($farthest);

        $rows = $this->sums->asArray();

        foreach ($dead as $t => $j) {
            if (!isset($order[$t])) {
                break;
            }

            $anchor = $samples[$order[$t]];

            $newRow = [];

            foreach ($anchor as $x) {
                $newRow[] = $massArr[$j] * $x;
            }

            $rows[$j] = $newRow;
        }

        $this->sums = Matrix::fromArray($rows, false);
    }

    /**
     * Re-derive the running centroids as the exact means of the members of
     * the closing pass, keeping the running positions of the clusters that
     * the pass left unoccupied.
     *
     * @internal
     *
     * @param Dataset $dataset
     * @param positive-int $n
     * @param list<positive-int> $rowCounts
     */
    protected function finalize(Dataset $dataset, int $n, array $rowCounts) : void
    {
        $closingRows = $rowCounts[count($rowCounts) - 1];

        $source = $closingRows < $n ? $dataset->randomSubset($closingRows) : $dataset;

        $d = $dataset->numFeatures();

        $sums = Matrix::zeros($this->k, $d);

        $counts = ColumnVector::zeros($this->k);

        foreach ($source->randomize()->batch($this->resolvedBatch) as $miniBatch) {
            $samples = $miniBatch->samples();

            $labels = $this->assignSamples($samples);

            $M = ColumnVector::fromArray(array_map('floatval', $labels), false)
                ->outer(Vector::ones($this->k))
                ->equal(Vector::range(0, $this->k - 1)->outer(Vector::ones(count($samples)))->transpose());

            $sums = $sums->addMatrix($M->transpose()->matmul(Matrix::fromArray($samples, false)));

            $counts = $counts->add($M->transpose()->sum());
        }

        $running = $this->sums->divideColumnVector($this->masses);

        $countsArr = $counts->asArray();

        $sumsArr = $sums->asArray();

        $runningArr = $running->asArray();

        $centroids = [];

        $sizes = [];

        foreach ($countsArr as $j => $count) {
            if ($count > 0.0) {
                $row = [];

                foreach ($sumsArr[$j] as $x) {
                    $row[] = $x / $count;
                }

                $centroids[] = $row;
            } else {
                $centroids[] = $runningArr[$j];
            }

            $sizes[] = (int) $count;
        }

        $this->centroids = $centroids;

        $this->sizes = $sizes;
    }

    /**
     * Compute the distance from a sample to each centroid.
     *
     * @internal
     *
     * @param list<int|float> $sample
     * @return list<float>
     */
    protected function centroidDistances(array $sample) : array
    {
        $distances = [];

        foreach ($this->centroids as $centroid) {
            $distances[] = $this->kernel->compute($sample, $centroid);
        }

        return $distances;
    }

    /**
     * Return an associative array containing the data used to serialize the object.
     *
     * @return mixed[]
     */
    public function __serialize() : array
    {
        $properties = get_object_vars($this);

        unset($properties['losses'], $properties['logger'], $properties['sums'], $properties['masses']);

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
        return 'E Means (' . Params::stringify($this->params()) . ')';
    }
}
