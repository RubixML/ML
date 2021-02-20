<?php

namespace Rubix\ML\Embedders;

use Tensor\Matrix;
use Rubix\ML\Verbose;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Other\Helpers\Params;
use Rubix\ML\Transformers\Transformer;
use Rubix\ML\Other\Traits\LoggerAware;
use Rubix\ML\Kernels\Distance\Distance;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\Specifications\SamplesAreCompatibleWithTransformer;
use Rubix\ML\Exceptions\InvalidArgumentException;

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
class TSNE implements Embedder, Verbose
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
     * @var int
     */
    protected $dimensions;

    /**
     * The number of degrees of freedom for the student's t distribution.
     *
     * @var int
     */
    protected $dofs;

    /**
     * The precomputed c factor of the gradient computation.
     *
     * @var float
     */
    protected $c;

    /**
     * The learning rate that controls the global step size.
     *
     * @var float
     */
    protected $rate;

    /**
     * The number of effective nearest neighbors to refer to when computing
     * the variance of the distribution over that sample.
     *
     * @var float
     */
    protected $perplexity;

    /**
     * The desired entropy of the Gaussian component over each sample i.e the
     * log perplexity.
     *
     * @var float
     */
    protected $entropy;

    /**
     * The factor to exaggerate the distances between samples by during the
     * early stage of fitting.
     *
     * @var float
     */
    protected $exaggeration;

    /**
     * The number of times to iterate over the embedding.
     *
     * @var int
     */
    protected $epochs;

    /**
     * The number of epochs that are considered to be in the early training
     * stage.
     *
     * @var int
     */
    protected $early;

    /**
     * The minimum norm of the gradient necessary to continue embedding.
     *
     * @var float
     */
    protected $minGradient;

    /**
     * The number of epochs without improvement in the training loss to wait
     * before considering an early stop.
     *
     * @var int
     */
    protected $window;

    /**
     * The distance metric used to measure distances between samples in both
     * high and low dimensions.
     *
     * @var \Rubix\ML\Kernels\Distance\Distance
     */
    protected $kernel;

    /**
     * The loss at each epoch from the last embedding.
     *
     * @var float[]|null
     */
    protected $steps;

    /**
     * @param int $dimensions
     * @param float $rate
     * @param int $perplexity
     * @param float $exaggeration
     * @param int $epochs
     * @param float $minGradient
     * @param int $window
     * @param \Rubix\ML\Kernels\Distance\Distance|null $kernel
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function __construct(
        int $dimensions = 2,
        float $rate = 100.0,
        int $perplexity = 30,
        float $exaggeration = 12.0,
        int $epochs = 1000,
        float $minGradient = 1e-7,
        int $window = 5,
        ?Distance $kernel = null
    ) {
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

        if ($window < 1) {
            throw new InvalidArgumentException('Window must be'
                . " greater than 0, $window given.");
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
        $this->window = $window;
        $this->kernel = $kernel ?? new Euclidean();
    }

    /**
     * Return the data types that this transformer is compatible with.
     *
     * @internal
     *
     * @return list<\Rubix\ML\DataType>
     */
    public function compatibility() : array
    {
        return $this->kernel->compatibility();
    }

    /**
     * Return the settings of the hyper-parameters in an associative array.
     *
     * @internal
     *
     * @return mixed[]
     */
    public function params() : array
    {
        return [
            'dimensions' => $this->dimensions,
            'rate' => $this->rate,
            'perplexity' => $this->perplexity,
            'exaggeration' => $this->exaggeration,
            'epochs' => $this->epochs,
            'min_gradient' => $this->minGradient,
            'window' => $this->window,
            'kernel' => $this->kernel,
        ];
    }

    /**
     * Return the magnitudes of the gradient at each epoch from the last embedding.
     *
     * @return float[]|null
     */
    public function steps() : ?array
    {
        return $this->steps;
    }

    /**
     * Transform the dataset in place.
     *
     * @param array[] $samples
     */
    public function transform(array &$samples) : void
    {
        SamplesAreCompatibleWithTransformer::with(new Unlabeled($samples), $this)->check();

        if ($this->logger) {
            $this->logger->info("$this initialized");

            $this->logger->info('Computing high-dimensional affinities');
        }

        $m = count($samples);

        $distances = $this->pairwiseDistances($samples);

        $p = Matrix::quick($this->affinities($distances))
            ->multiply($this->exaggeration);

        $y = Matrix::gaussian($m, $this->dimensions)
            ->multiply(self::Y_INIT_SCALE);

        $velocity = Matrix::zeros($m, $this->dimensions);
        $gains = Matrix::ones($m, $this->dimensions)->asArray();

        $momentum = self::INIT_MOMENTUM;
        $bestLoss = INF;
        $delta = 0;

        $this->steps = [];

        for ($epoch = 1; $epoch <= $this->epochs; ++$epoch) {
            $distances = $this->pairwiseDistances($y->asArray());

            $gradient = $this->gradient($p, $y, Matrix::quick($distances));

            $directions = $velocity->multiply($gradient)->asArray();

            foreach ($gains as $i => &$row) {
                $row = array_map([$this, 'attenuate'], $row, $directions[$i]);
            }

            unset($row);

            $gradient = $gradient->multiply(Matrix::quick($gains));

            $velocity = $velocity->multiply($momentum)
                ->subtract($gradient->multiply($this->rate));

            $y = $y->add($velocity);

            $loss = $gradient->l2Norm();

            if (is_nan($loss)) {
                if ($this->logger) {
                    $this->logger->info('Numerical instability detected');
                }

                break;
            }

            $this->steps[] = $loss;

            if ($this->logger) {
                $this->logger->info("Epoch $epoch - Gradient: $loss");
            }

            if ($loss < $this->minGradient) {
                break;
            }

            if ($loss < $bestLoss) {
                $bestLoss = $loss;

                $delta = 0;
            } else {
                ++$delta;
            }

            if ($delta >= $this->window) {
                break;
            }

            if ($epoch === $this->early) {
                $p = $p->divide($this->exaggeration);

                $momentum += self::MOMENTUM_BOOST;

                if ($this->logger) {
                    $this->logger->info('Early exaggeration exhausted');
                }
            }
        }

        if ($this->logger) {
            $this->logger->info('Embedding complete');
        }

        $samples = $y->asArray();
    }

    /**
     * Calculate the pairwise distances for each sample and return them in a 2-d array.
     *
     * @param array[] $samples
     * @return array[]
     */
    protected function pairwiseDistances(array $samples) : array
    {
        $distances = [];

        foreach ($samples as $i => $sampleA) {
            $row = [];

            foreach ($samples as $j => $sampleB) {
                $row[] = $i !== $j ? $this->kernel->compute($sampleA, $sampleB) : 0.0;
            }

            $distances[] = $row;
        }

        return $distances;
    }

    /**
     * Compute the conditional probabilities from the distance matrix such that
     * they approximately match the desired perplexity.
     *
     * @param array[] $distances
     * @return array[]
     */
    protected function affinities(array $distances) : array
    {
        $affinities = [];

        foreach ($distances as $i => $row) {
            $candidate = [];
            $maxBeta = INF;
            $minBeta = -INF;
            $beta = 1.0;

            for ($j = 0; $j < self::MAX_BINARY_PRECISION; ++$j) {
                $candidate = [];
                $pSigma = 0.0;

                foreach ($row as $k => $distance) {
                    if ($i !== $k) {
                        $affinity = exp(-$distance * $beta);

                        $candidate[] = $affinity;
                        $pSigma += $affinity;
                    } else {
                        $candidate[] = 0.0;
                    }
                }

                $pSigma = $pSigma ?: EPSILON;

                $distSigma = 0.0;

                foreach ($candidate as $k => &$affinity) {
                    $affinity /= $pSigma;

                    $distSigma += $row[$k] * $affinity;
                }

                $entropy = log($pSigma) + $beta * $distSigma;

                $diff = $this->entropy - $entropy;

                if (abs($diff) < self::PERPLEXITY_TOLERANCE) {
                    break;
                }

                if ($diff < 0.0) {
                    $minBeta = $beta;

                    if ($maxBeta === INF) {
                        $beta *= 2.0;
                    } else {
                        $beta = 0.5 * ($beta + $maxBeta);
                    }
                } else {
                    $maxBeta = $beta;

                    if ($minBeta === -INF) {
                        $beta /= 2.0;
                    } else {
                        $beta = 0.5 * ($beta + $minBeta);
                    }
                }
            }

            $affinities[] = $candidate;
        }

        return $affinities;
    }

    /**
     * Compute the gradient of the KL Divergence cost function with respect to the embedding.
     *
     * @param \Tensor\Matrix $p
     * @param \Tensor\Matrix $y
     * @param \Tensor\Matrix $distances
     * @return \Tensor\Matrix
     */
    protected function gradient(Matrix $p, Matrix $y, Matrix $distances) : Matrix
    {
        $q = $distances->divide($this->dofs)
            ->add(1.0)
            ->pow((1.0 + $this->dofs) / -2.0);

        $q = $q->divide($q->sum()->multiply(2.0))
            ->clipLower(EPSILON);

        $pqd = $p->subtract($q)->multiply($distances);

        $gradient = [];

        foreach ($pqd->asVectors() as $i => $vector) {
            $yHat = $y->rowAsVector($i)->subtract($y);

            $gradient[] = $vector->matmul($yHat)->row(0);
        }

        return Matrix::quick($gradient)
            ->multiply($this->c);
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
     * @return string
     */
    public function __toString() : string
    {
        return 't-SNE (' . Params::stringify($this->params()) . ')';
    }
}
