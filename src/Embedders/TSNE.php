<?php

namespace Rubix\ML\Embedders;

use Rubix\ML\Verbose;
use Rubix\Tensor\Matrix;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Other\Helpers\Params;
use Rubix\ML\Other\Helpers\DataType;
use Rubix\ML\Other\Traits\LoggerAware;
use Rubix\ML\Kernels\Distance\Distance;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\Other\Specifications\DatasetIsCompatibleWithEmbedder;
use InvalidArgumentException;

use const Rubix\ML\EPSILON;

/**
 * t-SNE
 *
 * T-distributed Stochastic Neighbor Embedding is a two-stage non-linear
 * manifold learning algorithm based on batch Gradient Descent. During the first
 * stage (*early* stage) the samples are exaggerated to encourage distant
 * clusters. Since the t-SNE cost function (KL Divergence) has a rough gradient,
 * momentum is employed to help escape bad local minima.
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

    protected const INIT_MOMENTUM = 0.5;
    protected const MOMENTUM_BOOST = 0.3;

    protected const GAIN_ACCELERATE = 0.2;
    protected const GAIN_BRAKE = 0.8;
    protected const MIN_GAIN = 0.01;

    protected const BINARY_PRECISION = 100;
    protected const SEARCH_TOLERANCE = 1e-5;

    protected const Y_INIT = 1e-4;

    /**
     * The number of dimensions of the embedding.
     *
     * @var int
     */
    protected $dimensions;

    /**
     * The number of degrees of freedom for the student t distribution.
     *
     * @var int
     */
    protected $degrees;

    /**
     * The learning rate that controls the step size.
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
     * The desired entropy of the Gaussian over each sample i.e the log
     * perplexity.
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
     * The minimum gradient necessary to continue fitting.
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
     * The magnitudes of the gradient at each epoch since the last embedding.
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
        float $rate = 100.,
        int $perplexity = 30,
        float $exaggeration = 12.,
        int $epochs = 1000,
        float $minGradient = 1e-7,
        int $window = 10,
        ?Distance $kernel = null
    ) {
        if ($dimensions < 1) {
            throw new InvalidArgumentException('Cannot embed into less than 1'
                . " dimension, $dimensions given.");
        }

        if ($rate <= 0.) {
            throw new InvalidArgumentException('Learning rate must be greater'
                . " than 0, $rate given.");
        }

        if ($perplexity < 1) {
            throw new InvalidArgumentException('Perplexity cannot be less than'
                . " 1, $perplexity given.");
        }

        if ($exaggeration < 1.) {
            throw new InvalidArgumentException('Early exaggeration must be 1 or'
             . " greater, $exaggeration given.");
        }

        if ($epochs < 1) {
            throw new InvalidArgumentException('Estimator must train for at'
                . " least 1 epoch, $epochs given.");
        }

        if ($minGradient < 0.) {
            throw new InvalidArgumentException('The minimum magnitude of the'
                . " gradient must be 0 or greater, $minGradient given.");
        }

        if ($window < 1) {
            throw new InvalidArgumentException('Window must be at least 1'
                . " epoch, $window given.");
        }

        $this->dimensions = $dimensions;
        $this->degrees = max($dimensions - 1, 1);
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
     * @return int[]
     */
    public function compatibility() : array
    {
        return [
            DataType::CONTINUOUS,
        ];
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
     * @return array
     */
    public function embed(Dataset $dataset) : array
    {
        DatasetIsCompatibleWithEmbedder::check($dataset, $this);

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

        $x = Matrix::build($dataset->samples());

        if ($this->logger) {
            $this->logger->info('Computing pairwise affinities');
        }

        $distances = $this->pairwiseDistances($x);

        $p = $this->highAffinities($distances)
            ->multiply($this->exaggeration);

        $y = $yHat = Matrix::gaussian($dataset->numRows(), $this->dimensions)
            ->multiply(self::Y_INIT);

        $velocity = Matrix::zeros($dataset->numRows(), $this->dimensions);
        $gains = Matrix::ones($dataset->numRows(), $this->dimensions)->asArray();

        $momentum = self::INIT_MOMENTUM;
        $bestLoss = INF;
        $nu = 0;

        $this->steps = [];

        for ($epoch = 1; $epoch <= $this->epochs; $epoch++) {
            $distances = $this->pairwiseDistances($y);

            $gradient = $this->gradient($p, $y, $distances);

            $directions = $velocity->multiply($gradient);

            foreach ($gains as $i => &$row) {
                $direction = $directions[$i];

                foreach ($row as $j => &$gain) {
                    $gain = $direction[$j] < 0.
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
                $nu++;
            }

            if (is_nan($loss) or $loss < EPSILON) {
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
     * Calculate the pairwise distances for each sample.
     *
     * @param \Rubix\Tensor\Matrix $samples
     * @return \Rubix\Tensor\Matrix
     */
    protected function pairwiseDistances(Matrix $samples) : Matrix
    {
        $distances = [];

        foreach ($samples as $a) {
            $temp = [];

            foreach ($samples as $b) {
                $temp[] = $this->kernel->compute($a, $b);
            }

            $distances[] = $temp;
        }

        return Matrix::quick($distances);
    }

    /**
     * Calculate the joint likelihood of each sample in the high dimensional
     * space as being nearest neighbor to each other sample.
     *
     * @param \Rubix\Tensor\Matrix $distances
     * @return \Rubix\Tensor\Matrix
     */
    protected function highAffinities(Matrix $distances) : Matrix
    {
        $zeros = array_fill(0, count($distances), 0);

        $p = [];

        foreach ($distances as $i => $row) {
            $affinities = $zeros;
            $minBeta = -INF;
            $maxBeta = INF;
            $beta = 1.;

            for ($l = 0; $l < self::BINARY_PRECISION; $l++) {
                $affinities = [];
                $pSigma = 0.;

                foreach ($row as $j => $distance) {
                    if ($i !== $j) {
                        $affinity = exp(-$distance * $beta);
                    } else {
                        $affinity = 0.;
                    }

                    $affinities[] = $affinity;
                    $pSigma += $affinity;
                }

                if ($pSigma === 0.) {
                    $pSigma = EPSILON;
                }

                $distSigma = 0.;

                foreach ($affinities as $j => &$prob) {
                    $prob /= $pSigma;
                    $distSigma += $row[$j] * $prob;
                }

                $entropy = log($pSigma) + $beta * $distSigma;

                $diff = $entropy - $this->entropy;

                if (abs($diff) < self::SEARCH_TOLERANCE) {
                    break 1;
                }

                if ($diff > 0.) {
                    $minBeta = $beta;

                    if ($maxBeta === INF) {
                        $beta *= 2.;
                    } else {
                        $beta = ($beta + $maxBeta) / 2.;
                    }
                } else {
                    $maxBeta = $beta;

                    if ($minBeta === -INF) {
                        $beta /= 2.;
                    } else {
                        $beta = ($beta + $minBeta) / 2.;
                    }
                }
            }

            $p[] = $affinities;
        }

        $p = Matrix::quick($p);

        $pHat = $p->add($p->transpose());

        $sigma = $pHat->sum()->clipLower(EPSILON);

        return $pHat->divide($sigma);
    }

    /**
     * Compute the gradient of the KL Divergence cost function with respect to
     * the embedding.
     *
     * @param \Rubix\Tensor\Matrix $p
     * @param \Rubix\Tensor\Matrix $y
     * @param \Rubix\Tensor\Matrix $distances
     * @return \Rubix\Tensor\Matrix
     */
    protected function gradient(Matrix $p, Matrix $y, Matrix $distances) : Matrix
    {
        $q = $distances->square()
            ->divide($this->degrees)
            ->add(1.)
            ->pow((1. + $this->degrees) / -2.);

        $qSigma = $q->sum()->multiply(2.);

        $q = $q->divide($qSigma)->clipLower(EPSILON);

        $pqd = $p->subtract($q)->multiply($distances);

        $c = 2. * (1. + $this->degrees) / $this->degrees;

        $gradient = [];

        foreach ($pqd->asVectors() as $i => $row) {
            $yHat = $y->rowAsVector($i)->subtract($y);
            
            $gradient[] = $row->matmul($yHat)
                ->multiply($c)
                ->row(0);
        }

        return Matrix::quick($gradient);
    }
}
