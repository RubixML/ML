<?php

namespace Rubix\ML\Embedders;

use Rubix\ML\Verbose;
use Rubix\Tensor\Matrix;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Unlabeled;
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
     * The number of effective nearest neighbors to refer to when computing
     * the variance of the gaussian over that sample.
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
     * The learning rate.
     *
     * @var float
     */
    protected $rate;

    /**
     * The distance metric used to measure distances between samples in both
     * high and low dimensions.
     *
     * @var \Rubix\ML\Kernels\Distance\Distance
     */
    protected $kernel;

    /**
     * The minimum gradient necessary to continue fitting.
     *
     * @var float
     */
    protected $minGradient;

    /**
     * The number of most recent epochs to consider when determining an early
     * stop.
     *
     * @var int
     */
    protected $window;

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
     * @param int $perplexity
     * @param float $exaggeration
     * @param float $rate
     * @param \Rubix\ML\Kernels\Distance\Distance|null $kernel
     * @param int $epochs
     * @param float $minGradient
     * @param int $window
     * @throws \InvalidArgumentException
     */
    public function __construct(
        int $dimensions = 2,
        int $perplexity = 30,
        float $exaggeration = 12.,
        float $rate = 100.,
        ?Distance $kernel = null,
        int $epochs = 1000,
        float $minGradient = 1e-8,
        int $window = 3
    ) {
        if ($dimensions < 1) {
            throw new InvalidArgumentException('Cannot embed into less than 1'
                . " dimension, $dimensions given.");
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

        if ($rate <= 0.) {
            throw new InvalidArgumentException('Learning rate must be greater'
                . " than 0, $rate given.");
        }

        if ($minGradient < 0.) {
            throw new InvalidArgumentException('The minimum magnitude of the'
                . " gradient must be 0 or greater, $minGradient given.");
        }

        if ($window < 2) {
            throw new InvalidArgumentException('The window of epochs used for'
                . " monitoring must be greater than 1, $window given.");
        }

        $this->dimensions = $dimensions;
        $this->degrees = max($dimensions - 1, 1);
        $this->perplexity = $perplexity;
        $this->entropy = log($perplexity);
        $this->exaggeration = $exaggeration;
        $this->rate = $rate;
        $this->kernel = $kernel ?? new Euclidean();
        $this->epochs = $epochs;
        $this->early = max(250, (int) round($epochs / 4));
        $this->minGradient = $minGradient;
        $this->window = $window;
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
     * @return \Rubix\ML\Datasets\Dataset
     */
    public function embed(Dataset $dataset) : Dataset
    {
        DatasetIsCompatibleWithEmbedder::check($dataset, $this);

        if ($this->logger) {
            $this->logger->info('Embedder initialized w/ '
                . Params::stringify([
                    'dimensions' => $this->dimensions,
                    'perplexity' => $this->perplexity,
                    'exaggeration' => $this->exaggeration,
                    'rate' => $this->rate,
                    'kernel' => $this->kernel,
                    'epochs' => $this->epochs,
                    'min_gradient' => $this->minGradient,
                    'window' => $this->window,
                ]));
        }

        $n = $dataset->numRows();

        $x = Matrix::build($dataset->samples());

        if ($this->logger) {
            $this->logger->info('Computing affinity matrix');
        }

        $distances = $this->pairwiseDistances($x);

        $p = $this->highAffinities($distances)
            ->multiply($this->exaggeration);

        $y = $yHat = Matrix::gaussian($n, $this->dimensions)
            ->multiply(self::Y_INIT);

        $velocity = Matrix::zeros($n, $this->dimensions);
        $gains = Matrix::ones($n, $this->dimensions)->asArray();
        $momentum = self::INIT_MOMENTUM;

        $this->steps = [];

        for ($epoch = 1; $epoch <= $this->epochs; $epoch++) {
            $distances = $this->pairwiseDistances($y);

            $gradient = $this->gradient($p, $y, $distances);

            $magnitude = $gradient->l2Norm();

            $direction = $velocity->multiply($gradient);

            foreach ($gains as $i => &$row) {
                $temp = $direction[$i];

                foreach ($row as $j => &$gain) {
                    $gain = $temp[$j] < 0.
                        ? $gain + self::GAIN_ACCELERATE
                        : $gain * self::GAIN_BRAKE;

                    $gain = max(self::MIN_GAIN, $gain);
                }
            }

            $gradient = $gradient->multiply(Matrix::quick($gains));

            $velocity = $velocity->multiply($momentum)
                ->subtract($gradient->multiply($this->rate));

            $y = $y->add($velocity);

            $this->steps[] = $magnitude;

            if ($this->logger) {
                $this->logger->info("Epoch $epoch Gradient=$magnitude");
            }

            if (is_nan($magnitude)) {
                break 1;
            }

            if ($magnitude < $this->minGradient) {
                break 1;
            }

            if ($epoch > $this->window) {
                $window = array_slice($this->steps, -$this->window);

                $worst = $window;
                sort($worst);

                if ($window === $worst) {
                    break 1;
                }
            }

            if ($epoch === $this->early) {
                $p = $p->divide($this->exaggeration);

                $momentum += self::MOMENTUM_BOOST;

                if ($this->logger) {
                    $this->logger->info('Early exaggeration exhausted');
                }
            }
        }

        $samples = $y->asArray();

        if ($this->logger) {
            $this->logger->info('Embedding complete');
        }

        if ($dataset instanceof Labeled) {
            return Labeled::quick($samples, $dataset->labels());
        }

        return Unlabeled::quick($samples);
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
