<?php

namespace Rubix\ML\Transformers;

use NDArray;
use NumPower;
use Rubix\ML\DataType;
use Rubix\ML\Verbose;
use Rubix\ML\Helpers\Params;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Traits\LoggerAware;
use Rubix\ML\Specifications\ExtensionIsLoaded;
use Rubix\ML\Specifications\SpecificationChain;
use Rubix\ML\Specifications\ExtensionMinimumVersion;
use Rubix\ML\Specifications\SamplesAreCompatibleWithTransformer;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Generator;

use function count;

use const Rubix\ML\EPSILON;

/**
 * t-SNE
 *
 * *T-distributed Stochastic Neighbor Embedding* is a two-stage non-linear manifold
 * learning algorithm based on Batch Gradient Descent that seeks to maintain the
 * distances between samples in low-dimensional space. During the first stage (*early
 * stage*) the distances are exaggerated to encourage more pronounced clusters. Since
 * the t-SNE cost function (KL Divergence) has a rough gradient, momentum is employed
 * to help escape bad local minima.
 *
 * > **Note:** T-SNE is implemented using the *exact* method which scales quadratically
 * in the number of samples. Therefore, it is recommended to subsample datasets larger
 * than a few thousand samples.
 *
 * References:
 * [1] L. van der Maaten et al. (2008). Visualizing Data using t-SNE.
 * [2] L. van der Maaten. (2009). Learning a Parametric Embedding by Preserving
 * Local Structure.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class TSNE implements Transformer, Verbose
{
    use LoggerAware;

    /**
     * The maximum number of epochs with early exaggeration.
     *
     * @var int
     */
    protected const MAX_EARLY_EPOCHS = 250;

    /**
     * The initial momentum coefficient.
     *
     * @var float
     */
    protected const INIT_MOMENTUM = 0.5;

    /**
     * The amount of momentum added after the early exaggeration stage.
     *
     * @var float
     */
    protected const MOMENTUM_BOOST = 0.3;

    /**
     * The maximum number of binary search attempts.
     *
     * @var int
     */
    protected const MAX_BINARY_PRECISION = 100;

    /**
     * The amount of binary search error to tolerate.
     *
     * @var float
     */
    protected const PERPLEXITY_TOLERANCE = 1e-5;

    /**
     * The scaling coefficient of the initial embedding.
     *
     * @var float
     */
    protected const Y_INIT_SCALE = 1e-4;

    /**
     * The amount of gain to add while the direction of the gradient is the same.
     *
     * @var float
     */
    protected const GAIN_ACCELERATE = 0.2;

    /**
     * The amount of brake to apply when the direction of the gradient changes.
     *
     * @var float
     */
    protected const GAIN_BRAKE = 0.8;

    /**
     * The minimum amount of gain to apply at each update.
     *
     * @var float
     */
    protected const MIN_GAIN = 0.01;

    /**
     * The number of dimensions of the target embedding.
     *
     * @var positive-int
     */
    protected int $dimensions;

    /**
     * The number of degrees of freedom for the student's t distribution.
     *
     * @var int
     */
    protected int $dofs;

    /**
     * The precomputed c factor of the gradient computation.
     *
     * @var float
     */
    protected float $c;

    /**
     * The learning rate that controls the global step size.
     *
     * @var float
     */
    protected float $rate;

    /**
     * The number of effective nearest neighbors to refer to when computing
     * the variance of the distribution over that sample.
     *
     * @var float
     */
    protected float $perplexity;

    /**
     * The desired entropy of the Gaussian component over each sample i.e the log perplexity.
     *
     * @var float
     */
    protected float $entropy;

    /**
     * The factor to exaggerate the distances between samples by during the early stage of fitting.
     *
     * @var float
     */
    protected float $exaggeration;

    /**
     * The number of times to iterate over the embedding.
     *
     * @var int
     */
    protected int $epochs;

    /**
     * The number of epochs that are considered to be in the early training stage.
     *
     * @var int
     */
    protected int $early;

    /**
     * The minimum norm of the gradient necessary to continue embedding.
     *
     * @var float
     */
    protected float $minGradient;

    /**
     * The loss at each epoch from the last embedding.
     *
     * @var float[]|null
     */
    protected ?array $losses = null;

    /**
     * @param int $dimensions
     * @param float $rate
     * @param int $perplexity
     * @param float $exaggeration
     * @param int $epochs
     * @param float $minGradient
     * @throws InvalidArgumentException
     */
    public function __construct(
        int $dimensions = 2,
        float $rate = 100.0,
        int $perplexity = 30,
        float $exaggeration = 12.0,
        int $epochs = 1000,
        float $minGradient = 1e-7
    ) {
        SpecificationChain::with([
            new ExtensionIsLoaded('RubixNumPower'),
            new ExtensionMinimumVersion('RubixNumPower', '0.7.0'),
        ])->check();

        if ($dimensions < 1) {
            throw new InvalidArgumentException('Dimensions must be'
                . " greater than 0, $dimensions given.");
        }

        if ($rate <= 0.0) {
            throw new InvalidArgumentException('Learning rate must be'
                . " greater than 0, $rate given.");
        }

        if ($perplexity < 1) {
            throw new InvalidArgumentException('Perplexity must be'
                . " greater than 0, $perplexity given.");
        }

        if ($exaggeration < 1.0) {
            throw new InvalidArgumentException('Exaggeration must be'
             . " greater than 1, $exaggeration given.");
        }

        if ($epochs < 1) {
            throw new InvalidArgumentException('Number of epochs'
                . " must be greater than 0, $epochs given.");
        }

        if ($minGradient < 0.0) {
            throw new InvalidArgumentException('Minimum gradient must be'
                . " greater than 0, $minGradient given.");
        }

        $dofs = max($dimensions - 1, 1);

        $this->dimensions = $dimensions;
        $this->dofs = $dofs;
        $this->c = 2.0 * (1.0 + $dofs) / $dofs;
        $this->rate = $rate;
        $this->perplexity = $perplexity;
        $this->entropy = log($perplexity);
        $this->exaggeration = $exaggeration;
        $this->epochs = $epochs;
        $this->early = min(self::MAX_EARLY_EPOCHS, (int) round($epochs / 4));
        $this->minGradient = $minGradient;
    }

    /**
     * Return the data types that this transformer is compatible with.
     *
     * @internal
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
     * Return the magnitudes of the gradient at each epoch from the last embedding.
     *
     * @return float[]|null
     */
    public function losses() : ?array
    {
        return $this->losses;
    }

    /**
     * Transform the dataset in place.
     *
     * @param array<mixed[]> $samples
     */
    public function transform(array &$samples) : void
    {
        SamplesAreCompatibleWithTransformer::with(new Unlabeled($samples), $this)->check();

        if ($this->logger) {
            $this->logger->info('Computing high-dimensional affinities');
        }

        $m = count($samples);

        $distances = $this->pairwiseDistances(NumPower::array($samples, 'float32'));

        $p = NumPower::multiply($this->affinities($distances), $this->exaggeration);

        $y = NumPower::multiply(
            NumPower::standardNormal([$m, $this->dimensions]),
            self::Y_INIT_SCALE
        );

        $velocity = NumPower::zeros([$m, $this->dimensions], $y->dataType(), 0);
        $gains = NumPower::ones([$m, $this->dimensions], $y->dataType(), 0)->toArray();

        $momentum = self::INIT_MOMENTUM;

        $this->losses = [];

        for ($epoch = 1; $epoch <= $this->epochs; ++$epoch) {
            $distances = $this->pairwiseDistances($y);

            $gradient = $this->gradient($p, $y, $distances);

            $directions = NumPower::multiply($velocity, $gradient)->toArray();

            foreach ($gains as $i => &$row) {
                $row = array_map([$this, 'attenuate'], $row, $directions[$i]);
            }

            unset($row);

            $gradient = NumPower::multiply($gradient, NumPower::array($gains, $y->dataType()));

            $velocity = NumPower::subtract(
                NumPower::multiply($velocity, $momentum),
                NumPower::multiply($gradient, $this->rate)
            );

            $y = NumPower::add($y, $velocity);

            $loss = NumPower::sqrt(NumPower::sum(NumPower::square($gradient)));

            $this->losses[] = $loss;

            if ($this->logger) {
                $this->logger->info("Epoch: $epoch, Gradient: $loss");
            }

            if (is_nan($loss)) {
                if ($this->logger) {
                    $this->logger->warning('Numerical instability detected');
                }

                break;
            }

            if ($loss < $this->minGradient) {
                break;
            }

            if ($epoch === $this->early) {
                $p = NumPower::divide($p, $this->exaggeration);

                $momentum += self::MOMENTUM_BOOST;

                if ($this->logger) {
                    $this->logger->info('Early exaggeration stage exhausted');
                }
            }
        }

        if ($this->logger) {
            $this->logger->info('Embedding complete');
        }

        $samples = $y->toArray();
    }

    /**
     * Calculate the squared pairwise distances for each sample using the
     * ||a - b||^2 = ||a||^2 + ||b||^2 - 2a.b identity and return them as a
     * matrix.
     *
     * @param NDArray $samples
     * @return NDArray
     */
    protected function pairwiseDistances(NDArray $samples) : NDArray
    {
        $norms = NumPower::sum(NumPower::square($samples), axis: 1);

        $dots = NumPower::matmul($samples, NumPower::transpose($samples, [1, 0]));

        $result = NumPower::add(
            NumPower::multiply($dots, -2.0),
            NumPower::reshape($norms, [$samples->shape()[0], 1])
        );

        return NumPower::add($result, $norms);
    }

    /**
     * Compute the joint probabilities from the squared distance matrix such
     * that they approximately match the desired perplexity. The resulting
     * matrix is symmetric and globally normalized (total sum equals 1).
     *
     * @param NDArray $distances
     * @return NDArray
     */
    protected function affinities(NDArray $distances) : NDArray
    {
        $m = $distances->shape()[0];

        if ($m === 0) {
            return NumPower::array([], $distances->dataType());
        }

        $mask = NumPower::subtract(
            NumPower::ones([$m, $m], $distances->dataType(), 0),
            NumPower::identity($m, $distances->dataType(), 0)
        );

        $betas = array_fill(0, $m, 1.0);
        $minBetas = array_fill(0, $m, -INF);
        $maxBetas = array_fill(0, $m, INF);

        $converged = array_fill(0, $m, false);

        $active = $m;

        $candidate = NumPower::zeros([$m, $m], $distances->dataType(), 0);

        for ($j = 0; $j < self::MAX_BINARY_PRECISION; ++$j) {
            if ($active === 0) {
                break;
            }

            $betasColumn = NumPower::reshape(NumPower::array($betas, $distances->dataType()), [$m, 1]);

            $candidate = NumPower::multiply(
                NumPower::exp(
                    NumPower::negative(
                        NumPower::multiply($distances, $betasColumn)
                    )
                ),
                $mask
            );

            $sigma = NumPower::sum($candidate, axis: 1);

            $sigma = NumPower::add(
                $sigma,
                NumPower::multiply(NumPower::equal($sigma, 0.0), EPSILON)
            );

            $candidate = NumPower::divide($candidate, NumPower::reshape($sigma, [$m, 1]));

            $dcb = NumPower::multiply(
                NumPower::sum(NumPower::multiply($distances, $candidate), axis: 1),
                NumPower::array($betas, $distances->dataType())
            );

            $diff = NumPower::negative(
                NumPower::subtract(
                    NumPower::add(NumPower::log($sigma), $dcb),
                    $this->entropy
                )
            )->toArray();

            for ($i = 0; $i < $m; ++$i) {
                if ($converged[$i]) {
                    continue;
                }

                if (abs($diff[$i]) < self::PERPLEXITY_TOLERANCE) {
                    $converged[$i] = true;

                    --$active;

                    continue;
                }

                if ($diff[$i] < 0.0) {
                    $minBetas[$i] = $betas[$i];

                    $betas[$i] = $maxBetas[$i] === INF
                        ? $betas[$i] * 2.0
                        : 0.5 * ($betas[$i] + $maxBetas[$i]);
                } else {
                    $maxBetas[$i] = $betas[$i];

                    $betas[$i] = $minBetas[$i] === -INF
                        ? $betas[$i] / 2.0
                        : 0.5 * ($betas[$i] + $minBetas[$i]);
                }
            }
        }

        $scale = 1.0 / (2.0 * $m);

        return NumPower::multiply(
            NumPower::add($candidate, NumPower::transpose($candidate, [1, 0])),
            $scale
        );
    }

    /**
     * Compute the gradient of the KL Divergence cost function with respect
     * to the embedding.
     *
     * @param NDArray $p
     * @param NDArray $y
     * @param NDArray $distances
     * @return NDArray
     */
    protected function gradient(NDArray $p, NDArray $y, NDArray $distances) : NDArray
    {
        $base = NumPower::add(NumPower::divide($distances, $this->dofs), 1.0);

        $weights = NumPower::reciprocal($base);

        $kernel = $this->dofs === 1
            ? $weights
            : NumPower::pow($weights, (1.0 + $this->dofs) * 0.5);

        $norm = NumPower::sum($kernel) - NumPower::trace($kernel);

        $q = NumPower::divide($kernel, max($norm, EPSILON));

        $pqd = NumPower::multiply(NumPower::subtract($p, $q), $weights);

        $pqdSum = NumPower::reshape(
            NumPower::sum($pqd, axis: 1),
            [$y->shape()[0], 1]
        );

        return NumPower::multiply(
            NumPower::subtract(
                NumPower::multiply($y, $pqdSum),
                NumPower::matmul($pqd, $y)
            ),
            $this->c
        );
    }

    /**
     * Attenuate the momentum signal.
     *
     * @param float $gain
     * @param float $direction
     * @return float
     */
    protected function attenuate(float $gain, float $direction) : float
    {
        $value = $direction < 0.0
            ? $gain + self::GAIN_ACCELERATE
            : $gain * self::GAIN_BRAKE;

        return max(self::MIN_GAIN, $value);
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
        return 't-SNE (' . Params::stringify([
            'dimensions' => $this->dimensions,
            'rate' => $this->rate,
            'perplexity' => $this->perplexity,
            'exaggeration' => $this->exaggeration,
            'epochs' => $this->epochs,
            'min gradient' => $this->minGradient,
        ]) . ')';
    }
}
