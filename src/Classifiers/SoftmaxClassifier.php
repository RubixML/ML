<?php

namespace Rubix\ML\Classifiers;

use Rubix\ML\Online;
use Rubix\ML\Verbose;
use Rubix\Tensor\Matrix;
use Rubix\ML\Persistable;
use Rubix\ML\Probabilistic;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\DataFrame;
use Rubix\ML\NeuralNet\FeedForward;
use Rubix\ML\Other\Functions\Argmax;
use Rubix\ML\Other\Traits\LoggerAware;
use Rubix\ML\NeuralNet\Optimizers\Adam;
use Rubix\ML\NeuralNet\Layers\Multiclass;
use Rubix\ML\NeuralNet\Layers\Placeholder;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use Rubix\ML\NeuralNet\CostFunctions\CostFunction;
use Rubix\ML\NeuralNet\CostFunctions\CrossEntropy;
use InvalidArgumentException;
use RuntimeException;

/**
 * Softmax Classifier
 *
 * A generalization of logistic regression for multiple class outcomes using a
 * single layer neural network with a softmax output layer.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class SoftmaxClassifier implements Online, Probabilistic, Verbose, Persistable
{
    use LoggerAware;

    /**
     * The number of training samples to consider per iteration of gradient
     * descent.
     *
     * @var int
     */
    protected $batchSize;

    /**
     * The gradient descent optimizer.
     *
     * @var \Rubix\ML\NeuralNet\Optimizers\Optimizer
     */
    protected $optimizer;

    /**
     * The L2 regularization parameter.
     *
     * @var float
     */
    protected $alpha;

    /**
     * The maximum number of training epochs. i.e. the number of times to iterate
     * over the entire training set.
     *
     * @var int
     */
    protected $epochs;

    /**
     * The minimum change in the weights necessary to continue training.
     *
     * @var float
     */
    protected $minChange;

    /**
     * The function that computes the cost of an erroneous activation during
     * training.
     *
     * @var \Rubix\ML\NeuralNet\CostFunctions\CostFunction
     */
    protected $costFn;

    /**
     * The unique class labels.
     *
     * @var array
     */
    protected $classes = [
        //
    ];

    /**
     * The underlying neural network instance.
     *
     * @var \Rubix\ML\NeuralNet\FeedForward|null
     */
    protected $network;

    /**
     * The average cost of a training sample at each epoch.
     *
     * @var array
     */
    protected $steps = [
        //
    ];

    /**
     * @param  int  $batchSize
     * @param  \Rubix\ML\NeuralNet\Optimizers\Optimizer|null  $optimizer
     * @param  float  $alpha
     * @param  int  $epochs
     * @param  float  $minChange
     * @param  \Rubix\ML\NeuralNet\CostFunctions\CostFunction|null  $costFn
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(int $batchSize = 50, ?Optimizer $optimizer = null, float $alpha = 1e-4,
                            int $epochs = 1000, float $minChange = 1e-4, ?CostFunction $costFn = null)
    {
        if ($batchSize < 1) {
            throw new InvalidArgumentException("Cannot have less than 1 sample"
                . " per batch, $batchSize given.");
        }

        if ($alpha < 0.) {
            throw new InvalidArgumentException("L2 regularization penalty must"
                . " be non-negative, $alpha given.");
        }

        if ($epochs < 1) {
            throw new InvalidArgumentException("Estimator must train for at"
                . " least 1 epoch, $epochs given.");
        }

        if ($minChange < 0.) {
            throw new InvalidArgumentException("Minimum change cannot be less"
                . " than 0, $minChange given.");
        }

        if (is_null($optimizer)) {
            $optimizer = new Adam();
        }

        if (is_null($costFn)) {
            $costFn = new CrossEntropy();
        }

        $this->batchSize = $batchSize;
        $this->optimizer = $optimizer;
        $this->alpha = $alpha;
        $this->epochs = $epochs;
        $this->minChange = $minChange;
        $this->costFn = $costFn;
    }

    /**
     * Return the integer encoded type of estimator this is.
     *
     * @return int
     */
    public function type() : int
    {
        return self::CLASSIFIER;
    }

    /**
     * Return the average cost at every epoch.
     *
     * @return array
     */
    public function steps() : array
    {
        return $this->steps;
    }

    /**
     * Return the underlying neural network instance or null if not trained.
     *
     * @return \Rubix\ML\NeuralNet\FeedForward|null
     */
    public function network() : ?FeedForward
    {
        return $this->network;
    }

    /**
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @throws \InvalidArgumentException
     * @return void
     */
    public function train(Dataset $dataset) : void
    {
        if (!$dataset instanceof Labeled) {
            throw new InvalidArgumentException('This estimator requires a'
                . ' labeled training set.');
        }

        $this->classes = $dataset->possibleOutcomes();

        $this->network = new FeedForward(
            new Placeholder($dataset->numColumns()),
            [],
            new Multiclass($this->classes, $this->alpha, $this->costFn),
            $this->optimizer
        );

        $this->steps = [];

        $this->partial($dataset);
    }

    /**
     * Perform mini-batch gradient descent with given optimizer over the training
     * set and update the input weights accordingly.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @throws \InvalidArgumentException
     * @return void
     */
    public function partial(Dataset $dataset) : void
    {
        if (!$dataset instanceof Labeled) {
            throw new InvalidArgumentException('This estimator requires a'
                . ' labeled training set.');
        }

        if (in_array(DataFrame::CATEGORICAL, $dataset->types())) {
            throw new InvalidArgumentException('This estimator only works with'
            . ' continuous features.');
        }

        if (is_null($this->network)) {
            $this->train($dataset);
            return;
        }

        if ($this->logger) $this->logger->info('Training started');

        $n = $dataset->numRows();

        $previous = INF;

        for ($epoch = 0; $epoch < $this->epochs; $epoch++) {
            $batches = $dataset->randomize()->batch($this->batchSize);

            $loss = 0.;

            foreach ($batches as $batch) {
                $loss += $this->network->roundtrip($batch);
            }

            $loss /= $n;

            $this->steps[] = $loss;

            if ($this->logger) $this->logger->info("Epoch $epoch"
            . " complete, loss: $loss");

            if (abs($previous - $loss) < $this->minChange) {
                break 1;
            }

            $previous = $loss;
        }

        if ($this->logger) $this->logger->info("Training complete");
    }

    /**
     * Feed a sample through the network and make a prediction based on the highest
     * activated output neuron.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @return array
     */
    public function predict(Dataset $dataset) : array
    {
        $predictions = [];

        foreach ($this->proba($dataset) as $joint) {
            $predictions[] = Argmax::compute($joint);
        }

        return $predictions;
    }

    /**
     * Output a vector of class probabilities per sample.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @throws \InvalidArgumentException
     * @throws \RuntimeException
     * @return array
     */
    public function proba(Dataset $dataset) : array
    {
        if (in_array(DataFrame::CATEGORICAL, $dataset->types())) {
            throw new InvalidArgumentException('This estimator only works with'
            . ' continuous features.');
        }

        if (is_null($this->network)) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        $samples = Matrix::quick($dataset->samples())->transpose();

        $probabilities = [];

        foreach ($this->network->infer($samples)->transpose() as $activations) {
            $probabilities[] = array_combine($this->classes, $activations);
        }

        return $probabilities;
    }
}
