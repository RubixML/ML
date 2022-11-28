<?php

namespace Rubix\ML\Classifiers;

use Rubix\ML\Online;
use Rubix\ML\Learner;
use Rubix\ML\Verbose;
use Rubix\ML\DataType;
use Rubix\ML\Encoding;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Probabilistic;
use Rubix\ML\EstimatorType;
use Rubix\ML\Helpers\Params;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Traits\LoggerAware;
use Rubix\ML\NeuralNet\Snapshot;
use Rubix\ML\NeuralNet\FeedForward;
use Rubix\ML\NeuralNet\Layers\Dense;
use Rubix\ML\NeuralNet\Layers\Hidden;
use Rubix\ML\Traits\AutotrackRevisions;
use Rubix\ML\NeuralNet\Optimizers\Adam;
use Rubix\ML\NeuralNet\Layers\Multiclass;
use Rubix\ML\CrossValidation\Metrics\FBeta;
use Rubix\ML\NeuralNet\Layers\Placeholder1D;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use Rubix\ML\NeuralNet\Initializers\Xavier1;
use Rubix\ML\CrossValidation\Metrics\Metric;
use Rubix\ML\Specifications\DatasetIsLabeled;
use Rubix\ML\Specifications\DatasetIsNotEmpty;
use Rubix\ML\Specifications\SpecificationChain;
use Rubix\ML\NeuralNet\CostFunctions\CrossEntropy;
use Rubix\ML\Specifications\DatasetHasDimensionality;
use Rubix\ML\NeuralNet\CostFunctions\ClassificationLoss;
use Rubix\ML\Specifications\LabelsAreCompatibleWithLearner;
use Rubix\ML\Specifications\EstimatorIsCompatibleWithMetric;
use Rubix\ML\Specifications\SamplesAreCompatibleWithEstimator;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use Generator;

use function is_nan;
use function count;
use function get_object_vars;

/**
 * Multilayer Perceptron
 *
 * A multiclass feed forward neural network classifier with user-defined hidden layers. The
 * Multilayer Perceptron is a deep learning model capable of forming higher-order feature
 * representations through layers of computation. In addition, the MLP features progress
 * monitoring which stops training when it can no longer make progress. It utilizes network
 * snapshotting to make sure that it always has the best model parameters even if progress
 * declined during training.
 *
 * References:
 * [1] G. E. Hinton. (1989). Connectionist learning procedures.
 * [2] L. Prechelt. (1997). Early Stopping - but when?
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class MultilayerPerceptron implements Estimator, Learner, Online, Probabilistic, Verbose, Persistable
{
    use AutotrackRevisions, LoggerAware;

    /**
     * An array composing the user-specified hidden layers of the network in order.
     *
     * @var \Rubix\ML\NeuralNet\Layers\Hidden[]
     */
    protected array $hiddenLayers;

    /**
     * The number of training samples to process at a time.
     *
     * @var positive-int
     */
    protected int $batchSize;

    /**
     * The gradient descent optimizer used to update the network parameters.
     *
     * @var \Rubix\ML\NeuralNet\Optimizers\Optimizer
     */
    protected \Rubix\ML\NeuralNet\Optimizers\Optimizer $optimizer;

    /**
     * The amount of L2 regularization applied to the weights of the output layer.
     *
     * @var float
     */
    protected float $l2Penalty;

    /**
     * The maximum number of training epochs. i.e. the number of times to iterate before terminating.
     *
     * @var int<0,max>
     */
    protected int $epochs;

    /**
     * The minimum change in the training loss necessary to continue training.
     *
     * @var float
     */
    protected float $minChange;

    /**
     * The number of epochs without improvement in the validation score to wait before considering an early stop.
     *
     * @var positive-int
     */
    protected int $window;

    /**
     * The proportion of training samples to use for validation and progress monitoring.
     *
     * @var float
     */
    protected float $holdOut;

    /**
     * The function that computes the loss associated with an erroneous activation during training.
     *
     * @var \Rubix\ML\NeuralNet\CostFunctions\ClassificationLoss
     */
    protected \Rubix\ML\NeuralNet\CostFunctions\ClassificationLoss $costFn;

    /**
     * The validation metric used to score the generalization performance of the model during training.
     *
     * @var \Rubix\ML\CrossValidation\Metrics\Metric
     */
    protected \Rubix\ML\CrossValidation\Metrics\Metric $metric;

    /**
     * The underlying neural network instance.
     *
     * @var \Rubix\ML\NeuralNet\FeedForward|null
     */
    protected ?\Rubix\ML\NeuralNet\FeedForward $network = null;

    /**
     * The unique class labels.
     *
     * @var string[]|null
     */
    protected ?array $classes = null;

    /**
     * The validation scores at each epoch from the last training session.
     *
     * @var float[]|null
     */
    protected ?array $scores = null;

    /**
     * The loss at each epoch from the last training session.
     *
     * @var float[]|null
     */
    protected ?array $losses = null;

    /**
     * @param \Rubix\ML\NeuralNet\Layers\Hidden[] $hiddenLayers
     * @param int $batchSize
     * @param \Rubix\ML\NeuralNet\Optimizers\Optimizer|null $optimizer
     * @param float $l2Penalty
     * @param int $epochs
     * @param float $minChange
     * @param int $window
     * @param float $holdOut
     * @param \Rubix\ML\NeuralNet\CostFunctions\ClassificationLoss|null $costFn
     * @param \Rubix\ML\CrossValidation\Metrics\Metric|null $metric
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function __construct(
        array $hiddenLayers = [],
        int $batchSize = 128,
        ?Optimizer $optimizer = null,
        float $l2Penalty = 1e-4,
        int $epochs = 1000,
        float $minChange = 1e-4,
        int $window = 5,
        float $holdOut = 0.1,
        ?ClassificationLoss $costFn = null,
        ?Metric $metric = null
    ) {
        foreach ($hiddenLayers as $layer) {
            if (!$layer instanceof Hidden) {
                throw new InvalidArgumentException('Hidden layer'
                    . ' must implement the Hidden interface.');
            }
        }

        if ($batchSize < 1) {
            throw new InvalidArgumentException('Batch size must be'
                . " greater than 0, $batchSize given.");
        }

        if ($l2Penalty < 0.0) {
            throw new InvalidArgumentException('L2 Penalty must be'
                . " greater than 0, $l2Penalty given.");
        }

        if ($epochs < 0) {
            throw new InvalidArgumentException('Number of epochs'
                . " must be greater than 0, $epochs given.");
        }

        if ($minChange < 0.0) {
            throw new InvalidArgumentException('Minimum change must be'
                . " greater than 0, $minChange given.");
        }

        if ($window < 1) {
            throw new InvalidArgumentException('Window must be'
                . " greater than 0, $window given.");
        }

        if ($holdOut < 0.0 or $holdOut > 0.5) {
            throw new InvalidArgumentException('Hold out ratio must be'
                . " between 0 and 0.5, $holdOut given.");
        }

        if ($metric) {
            EstimatorIsCompatibleWithMetric::with($this, $metric)->check();
        }

        $this->hiddenLayers = $hiddenLayers;
        $this->batchSize = $batchSize;
        $this->optimizer = $optimizer ?? new Adam();
        $this->l2Penalty = $l2Penalty;
        $this->epochs = $epochs;
        $this->minChange = $minChange;
        $this->window = $window;
        $this->holdOut = $holdOut;
        $this->costFn = $costFn ?? new CrossEntropy();
        $this->metric = $metric ?? new FBeta();
    }

    /**
     * Return the estimator type.
     *
     * @internal
     *
     * @return \Rubix\ML\EstimatorType
     */
    public function type() : EstimatorType
    {
        return EstimatorType::classifier();
    }

    /**
     * Return the data types that the estimator is compatible with.
     *
     * @internal
     *
     * @return list<\Rubix\ML\DataType>
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
     * @internal
     *
     * @return mixed[]
     */
    public function params() : array
    {
        return [
            'hidden layers' => $this->hiddenLayers,
            'batch size' => $this->batchSize,
            'optimizer' => $this->optimizer,
            'l2 penalty' => $this->l2Penalty,
            'epochs' => $this->epochs,
            'min change' => $this->minChange,
            'window' => $this->window,
            'hold out' => $this->holdOut,
            'cost fn' => $this->costFn,
            'metric' => $this->metric,
        ];
    }

    /**
     * Has the learner been trained?
     *
     * @return bool
     */
    public function trained() : bool
    {
        return $this->network and $this->classes;
    }

    /**
     * Return an iterable progress table with the steps from the last training session.
     *
     * @return \Generator<mixed[]>
     */
    public function steps() : Generator
    {
        if (!$this->losses) {
            return;
        }

        foreach ($this->losses as $epoch => $loss) {
            yield [
                'epoch' => $epoch,
                'score' => $this->scores[$epoch] ?? null,
                'loss' => $loss,
            ];
        }
    }

    /**
     * Return the validation score for each epoch from the last training session.
     *
     * @return float[]|null
     */
    public function scores() : ?array
    {
        return $this->scores;
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
     * Return the underlying neural network instance or null if not trained.
     *
     * @return \Rubix\ML\NeuralNet\FeedForward|null
     */
    public function network() : ?FeedForward
    {
        return $this->network;
    }

    /**
     * Train the learner with a dataset.
     *
     * @param \Rubix\ML\Datasets\Labeled $dataset
     */
    public function train(Dataset $dataset) : void
    {
        SpecificationChain::with([
            new DatasetIsLabeled($dataset),
            new DatasetIsNotEmpty($dataset),
            new LabelsAreCompatibleWithLearner($dataset, $this),
        ])->check();

        $classes = $dataset->possibleOutcomes();

        $hiddenLayers = $this->hiddenLayers;

        $hiddenLayers[] = new Dense(count($classes), $this->l2Penalty, true, new Xavier1());

        $this->network = new FeedForward(
            new Placeholder1D($dataset->numFeatures()),
            $hiddenLayers,
            new Multiclass($classes, $this->costFn),
            $this->optimizer
        );

        $this->network->initialize();

        $this->classes = $classes;

        $this->partial($dataset);
    }

    /**
     * Train the network using mini-batch gradient descent with backpropagation.
     *
     * @param \Rubix\ML\Datasets\Labeled $dataset
     */
    public function partial(Dataset $dataset) : void
    {
        if (!$this->network) {
            $this->train($dataset);

            return;
        }

        SpecificationChain::with([
            new DatasetIsLabeled($dataset),
            new DatasetIsNotEmpty($dataset),
            new SamplesAreCompatibleWithEstimator($dataset, $this),
            new LabelsAreCompatibleWithLearner($dataset, $this),
            new DatasetHasDimensionality($dataset, $this->network->input()->width()),
        ])->check();

        if ($this->logger) {
            $this->logger->info("Training $this");
        }

        [$testing, $training] = $dataset->stratifiedSplit($this->holdOut);

        [$minScore, $maxScore] = $this->metric->range()->list();

        $bestScore = $minScore;
        $bestEpoch = $numWorseEpochs = 0;
        $loss = 0.0;
        $snapshot = null;
        $prevLoss = INF;

        if ($testing->empty() and $this->logger) {
            $this->logger->notice('Insufficient validation data, '
                . 'some features are disabled');
        }

        $this->scores = $this->losses = [];

        for ($epoch = 1; $epoch <= $this->epochs; ++$epoch) {
            $batches = $training->randomize()->batch($this->batchSize);

            $loss = 0.0;

            foreach ($batches as $batch) {
                $loss += $this->network->roundtrip($batch);
            }

            $loss /= count($batches);

            $lossChange = abs($prevLoss - $loss);

            $this->losses[$epoch] = $loss;

            if (is_nan($loss)) {
                if ($this->logger) {
                    $this->logger->warning('Numerical instability detected');
                }

                break;
            }

            if (!$testing->empty()) {
                $predictions = $this->predict($testing);

                $score = $this->metric->score($predictions, $testing->labels());

                $this->scores[$epoch] = $score;
            }

            if ($this->logger) {
                $lossDirection = $loss < $prevLoss ? '↓' : '↑';

                $message = "Epoch: $epoch, "
                    . "{$this->costFn}: $loss, "
                    . "Loss Change: {$lossDirection}{$lossChange}, "
                    . "{$this->metric}: " . ($score ?? 'N/A');

                $this->logger->info($message);
            }

            if (isset($score)) {
                if ($score >= $maxScore) {
                    break;
                }

                if ($score > $bestScore) {
                    $bestScore = $score;
                    $bestEpoch = $epoch;

                    $snapshot = Snapshot::take($this->network);

                    $numWorseEpochs = 0;
                } else {
                    ++$numWorseEpochs;
                }

                if ($numWorseEpochs >= $this->window) {
                    break;
                }
            }

            if ($lossChange < $this->minChange) {
                break;
            }

            $prevLoss = $loss;
        }

        if ($snapshot and (end($this->scores) < $bestScore or is_nan($loss))) {
            $snapshot->restore();

            if ($this->logger) {
                $this->logger->info("Model state restored to epoch $bestEpoch");
            }
        }

        if ($this->logger) {
            $this->logger->info('Training complete');
        }
    }

    /**
     * Make predictions from a dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @return list<string>
     */
    public function predict(Dataset $dataset) : array
    {
        return array_map('Rubix\ML\argmax', $this->proba($dataset));
    }

    /**
     * Estimate the joint probabilities for each possible outcome.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \Rubix\ML\Exceptions\RuntimeException
     * @return list<array<string,float>>
     */
    public function proba(Dataset $dataset) : array
    {
        if (!$this->network or !$this->classes) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        DatasetHasDimensionality::with($dataset, $this->network->input()->width())->check();

        $activations = $this->network->infer($dataset);

        $probabilities = [];

        foreach ($activations->asArray() as $dist) {
            $probabilities[] = array_combine($this->classes, $dist) ?: [];
        }

        return $probabilities;
    }

    /**
     * Export the network architecture as a graph in dot format.
     *
     * @throws \Rubix\ML\Exceptions\RuntimeException
     * @return \Rubix\ML\Encoding
     */
    public function exportGraphviz() : Encoding
    {
        if (!$this->network) {
            throw new RuntimeException('Must train network first.');
        }

        return $this->network->exportGraphviz();
    }

    /**
     * Return an associative array containing the data used to serialize the object.
     *
     * @return mixed[]
     */
    public function __serialize() : array
    {
        $properties = get_object_vars($this);

        unset($properties['losses'], $properties['scores']);

        return $properties;
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
        return 'Multilayer Perceptron (' . Params::stringify($this->params()) . ')';
    }
}
