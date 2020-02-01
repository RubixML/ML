<?php

namespace Rubix\ML\Embedders;

use Tensor\Matrix;
use Rubix\ML\Verbose;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Other\Helpers\Params;
use Rubix\ML\Other\Traits\LoggerAware;
use Rubix\ML\Kernels\Distance\Distance;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\Other\Specifications\SamplesAreCompatibleWithEmbedder;
use InvalidArgumentException;

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
    protected const Y_INIT = 1e-4;

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
    protected $degrees;

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
     * The training loss at each epoch since the last embedding.
     *
     * @var float[]
     */
    protected $steps = [
        //
    ];

    /**
     * @param int $dimensions
     * @param float $rate
     * @param int $perplexity
     * @param float $exaggeration
     * @param int $epochs
     * @param float $minGradient
     * @param int $window
     * @param \Rubix\ML\Kernels\Distance\Distance|null $kernel
     * @throws \InvalidArgumentException
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
            throw new InvalidArgumentException('Cannot target less than 1'
                . " dimension, $dimensions given.");
        }

        if ($rate <= 0.0) {
            throw new InvalidArgumentException('Learning rate must be greater'
                . " than 0, $rate given.");
        }

        if ($perplexity < 1) {
            throw new InvalidArgumentException('Perplexity cannot be less'
                . " than 1, $perplexity given.");
        }

        if ($exaggeration < 1.0) {
            throw new InvalidArgumentException('Early exaggeration must be 1'
             . " or greater, $exaggeration given.");
        }

        if ($epochs < 1) {
            throw new InvalidArgumentException('Estimator must train for at'
                . " least 1 epoch, $epochs given.");
        }

        if ($minGradient < 0.0) {
            throw new InvalidArgumentException('Mminimum gradient must be'
                . " 0 or greater, $minGradient given.");
        }

        if ($window < 1) {
            throw new InvalidArgumentException('Window must be at least 1'
                . " epoch, $window given.");
        }

        $this->dimensions = $dimensions;
        $this->degrees = max($dimensions - 1, 1);
        $this->c = 2.0 * (1. + $this->degrees) / $this->degrees;
        $this->rate = $rate;
        $this->perplexity = $perplexity;
        $this->entropy = log($perplexity);
        $this->exaggeration = $exaggeration;
        $this->epochs = $epochs;
        $this->early = min(250, (int) round($epochs / 4));
        $this->minGradient = $minGradient;
        $this->window = $window;
        $this->kernel = $kernel ?? new Euclidean();
    }

    /**
     * Return the data types that this embedder is compatible with.
     *
     * @return \Rubix\ML\DataType[]
     */
    public function compatibility() : array
    {
        return $this->kernel->compatibility();
    }

    /**
     * Return the magnitudes of the gradient at each epoch from the last
     * embedding.
     *
     * @return float[]
     */
    public function steps() : array
    {
        return $this->steps;
    }

    /**
     * Embed a high dimensional dataset into a lower dimensional one.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \InvalidArgumentException
     * @return array[]
     */
    public function embed(Dataset $dataset) : array
    {
        SamplesAreCompatibleWithEmbedder::check($dataset, $this);

        if ($this->logger) {
            $this->logger->info('Embedder init w/ ' . Params::stringify([
                'dimensions' => $this->dimensions,
                'rate' => $this->rate,
                'perplexity' => $this->perplexity,
                'exaggeration' => $this->exaggeration,
                'epochs' => $this->epochs,
                'min_gradient' => $this->minGradient,
                'window' => $this->window,
                'kernel' => $this->kernel,
            ]));
        }

        $m = $dataset->numRows();

        if ($this->logger) {
            $this->logger->info('Computing high-dimensional'
                . ' pairwise affinities');
        }

        $distances = $this->pairwiseDistances($dataset->samples());

        $p = $this->affinities($distances)
            ->multiply($this->exaggeration);

        $y = $yHat = Matrix::gaussian($m, $this->dimensions)
            ->multiply(self::Y_INIT);

        $velocity = Matrix::zeros($m, $this->dimensions);
        $gains = Matrix::ones($m, $this->dimensions)->asArray();

        $momentum = self::INIT_MOMENTUM;
        $bestLoss = INF;
        $nu = 0;

        $this->steps = [];

        for ($epoch = 1; $epoch <= $this->epochs; ++$epoch) {
            $distances = $this->pairwiseDistances($y->asArray());

            $gradient = $this->gradient($p, $y, Matrix::quick($distances));

            $directions = $velocity->multiply($gradient)->asArray();

            foreach ($gains as $i => &$row) {
                $direction = $directions[$i];

                foreach ($row as $j => &$gain) {
                    $gain = $direction[$j] < 0.0
                        ? $gain + self::GAIN_ACCELERATE
                        : $gain * self::GAIN_BRAKE;

                    $gain = max(self::MIN_GAIN, $gain);
                }
            }

            $gradient = $gradient->multiply(Matrix::quick($gains));

            $velocity = $velocity->multiply($momentum)
                ->subtract($gradient->multiply($this->rate));

            $y = $y->add($velocity);

            $loss = $gradient->l2Norm();

            $this->steps[] = $loss;

            if ($this->logger) {
                $this->logger->info("Epoch $epoch loss=$loss");
            }

            if ($loss < $bestLoss) {
                $bestLoss = $loss;
                
                $nu = 0;
            } else {
                ++$nu;
            }

            if (is_nan($loss)) {
                break 1;
            }

            if ($loss < $this->minGradient) {
                break 1;
            }

            if ($nu >= $this->window) {
                break 1;
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

        return $y->asArray();
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
                if ($i !== $j) {
                    $row[] = $this->kernel->compute($sampleA, $sampleB);
                } else {
                    $row[] = 0.0;
                }
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
     * @return \Tensor\Matrix
     */
    protected function affinities(array $distances) : Matrix
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
                    break 1;
                }

                if ($diff < 0.0) {
                    $minBeta = $beta;

                    if ($maxBeta === INF) {
                        $beta *= 2.0;
                    } else {
                        $beta = ($beta + $maxBeta) / 2.0;
                    }
                } else {
                    $maxBeta = $beta;

                    if ($minBeta === -INF) {
                        $beta /= 2.0;
                    } else {
                        $beta = ($beta + $minBeta) / 2.0;
                    }
                }
            }

            $affinities[] = $candidate;
        }

        return Matrix::quick($affinities);
    }

    /**
     * Compute the gradient of the KL Divergence cost function with respect to
     * the embedding.
     *
     * @param \Tensor\Matrix $p
     * @param \Tensor\Matrix $y
     * @param \Tensor\Matrix $distances
     * @return \Tensor\Matrix
     */
    protected function gradient(Matrix $p, Matrix $y, Matrix $distances) : Matrix
    {
        $q = $distances->divide($this->degrees)
            ->add(1.0)
            ->pow((1.0 + $this->degrees) / -2.0);

        $q = $q->divide($q->sum()->multiply(2.0))
            ->clipLower(EPSILON);

        $pqd = $p->subtract($q)->multiply($distances);

        $gradient = [];

        foreach ($pqd->asVectors() as $i => $row) {
            $yHat = $y->rowAsVector($i)->subtract($y);
            
            $gradient[] = $row->matmul($yHat)->row(0);
        }

        return Matrix::quick($gradient)
            ->multiply($this->c);
    }
}
