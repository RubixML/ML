<?php

namespace Rubix\ML\Classifiers;

use Generator;
use Rubix\ML\Online;
use Rubix\ML\Learner;
use Rubix\ML\Verbose;
use Rubix\ML\DataType;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Probabilistic;
use Rubix\ML\RanksFeatures;
use Rubix\ML\EstimatorType;
use Rubix\ML\Helpers\Params;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Traits\LoggerAware;
use Rubix\ML\NeuralNet\Network;
use Rubix\ML\NeuralNet\Snapshot;
use Rubix\ML\NeuralNet\Layers\Binary;
use Rubix\ML\NeuralNet\Layers\Dense;
use Rubix\ML\NeuralNet\Initializers\Xavier1;
use Rubix\ML\Traits\AutotrackRevisions;
use Rubix\ML\NeuralNet\FeedForward;
use Rubix\ML\NeuralNet\Layers\Placeholder1D;
use Rubix\ML\NeuralNet\Optimizers\Adam;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use Rubix\ML\CrossValidation\Metrics\FBeta;
use Rubix\ML\CrossValidation\Metrics\Metric;
use Rubix\ML\Specifications\DatasetIsLabeled;
use Rubix\ML\Specifications\DatasetIsNotEmpty;
use Rubix\ML\Specifications\SpecificationChain;
use Rubix\ML\NeuralNet\CostFunctions\BinaryCrossEntropy;
use Rubix\ML\Specifications\DatasetHasDimensionality;
use Rubix\ML\NeuralNet\CostFunctions\ClassificationLoss;
use Rubix\ML\Specifications\LabelsAreCompatibleWithLearner;
use Rubix\ML\Specifications\SamplesAreCompatibleWithEstimator;
use Rubix\ML\Specifications\EstimatorIsCompatibleWithMetric;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;

use function is_nan;
use function is_dir;
use function count;
use function uniqid;
use function get_object_vars;
use function number_format;
use function array_map;
use function sys_get_temp_dir;

/**
 * Logistic Regression
 *
 * A linear classifier that uses the logistic (*sigmoid*) function to estimate the
 * probabilities of exactly two class outcomes. The model parameters (weights and bias)
 * are solved using Mini Batch Gradient Descent with pluggable optimizers and cost
 * functions that run on the neural network subsystem.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class LogisticRegression implements Estimator, Learner, Online, Probabilistic, RanksFeatures, Verbose, Persistable
{
    use AutotrackRevisions, LoggerAware;

    /**
     * The number of training samples to process at a time.
     *
     * @var positive-int
     */
    protected int $batchSize;

    /**
     * The gradient descent optimizer used to update the network parameters.
     *
     * @var Optimizer
     */
    protected Optimizer $optimizer;

    /**
     * The amount of L2 regularization applied to the weights of the output layer.
     *
     * @var float
     */
    protected float $l2Penalty;

    /**
     * The maximum number of training epochs. i.e. the number of times to iterate before terminating.
     *
     * @var int
     */
    protected int $epochs;

    /**
     * The minimum change in the training loss necessary to continue training.
     *
     * @var float
     */
    protected float $minChange;

    /**
     * The number of epochs to train before evaluating the model with the holdout set.
     *
     * @var int
     */
    protected int $evalInterval;

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
     * @var ClassificationLoss
     */
    protected ClassificationLoss $costFn;

    /**
     * The validation metric used to score the generalization performance of the model during training.
     *
     * @var Metric
     */
    protected Metric $metric;

    /**
     * The underlying neural network instance.
     *
     * @var FeedForward|null
     */
    protected ?FeedForward $network = null;

    /**
     * The unique class labels.
     *
     * @var string[]|null
     */
    protected ?array $classes = null;

    /**
     * The loss at each epoch from the last training session.
     *
     * @var float[]|null
     */
    protected ?array $losses = null;

    /**
     * The validation scores at each epoch from the last training session.
     *
     * @var float[]|null
     */
    protected ?array $scores = null;

    /**
     * The file path to store the snapshot on disk during training.
     *
     * @var string|null
     */
    protected ?string $snapshotPath = null;

    /**
     * @param int $batchSize
     * @param Optimizer|null $optimizer
     * @param float $l2Penalty
     * @param int $epochs
     * @param float $minChange
     * @param int $evalInterval
     * @param int $window
     * @param float $holdOut
     * @param ClassificationLoss|null $costFn
     * @param Metric|null $metric
     * @throws InvalidArgumentException
     */
    public function __construct(
        int $batchSize = 128,
        ?Optimizer $optimizer = null,
        float $l2Penalty = 1e-4,
        int $epochs = 1000,
        float $minChange = 1e-4,
        int $evalInterval = 3,
        int $window = 5,
        float $holdOut = 0.1,
        ?ClassificationLoss $costFn = null,
        ?Metric $metric = null
    ) {
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

        if ($evalInterval < 1) {
            throw new InvalidArgumentException('Eval interval must be'
                . " greater than 0, $evalInterval given.");
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

        $this->batchSize = $batchSize;
        $this->optimizer = $optimizer ?? new Adam();
        $this->l2Penalty = $l2Penalty;
        $this->epochs = $epochs;
        $this->minChange = $minChange;
        $this->evalInterval = $evalInterval;
        $this->window = $window;
        $this->holdOut = $holdOut;
        $this->costFn = $costFn ?? new BinaryCrossEntropy();
        $this->metric = $metric ?? new FBeta();
    }

    /**
     * Return the estimator type.
     *
     * @internal
     *
     * @return EstimatorType
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
     * @internal
     *
     * @return mixed[]
     */
    public function params() : array
    {
        return [
            'batch size' => $this->batchSize,
            'optimizer' => $this->optimizer,
            'l2 penalty' => $this->l2Penalty,
            'epochs' => $this->epochs,
            'min change' => $this->minChange,
            'eval interval' => $this->evalInterval,
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
                'score' => $this->scores[$epoch] ?? null,
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
     * Return the validation score at each epoch from the last training session.
     *
     * @return float[]|null
     */
    public function scores() : ?array
    {
        return $this->scores;
    }

    /**
     * Return the underlying neural network instance or null if not trained.
     *
     * @return Network|null
     */
    public function network() : ?Network
    {
        return $this->network;
    }

    /**
     * Set the file path to store the snapshot on disk during training.
     *
     * @param string|null $path
     * @throws InvalidArgumentException
     */
    public function setSnapshotPath(?string $path) : void
    {
        if (isset($path) and is_dir($path)) {
            throw new InvalidArgumentException('Snapshot path must be to a file, folder given.');
        }

        $this->snapshotPath = $path;
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

        $this->network = new FeedForward(
            new Placeholder1D($dataset->numFeatures()),
            [new Dense(1, $this->l2Penalty, true, new Xavier1())],
            new Binary($classes, $this->costFn),
            $this->optimizer
        );

        $this->network->initialize();

        $this->classes = $classes;

        $this->partial($dataset);
    }

    /**
     * Perform a partial train on the learner.
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

            $numParams = number_format($this->network->numParams());

            $this->logger->info("{$numParams} trainable parameters");
        }

        [$testing, $training] = $dataset->stratifiedSplit($this->holdOut);

        [$minScore, $maxScore] = $this->metric->range()->list();

        $bestScore = $minScore;
        $bestEpoch = $numWorseEpochs = 0;
        $loss = 0.0;
        $score = $snapshot = null;
        $prevLoss = INF;

        $snapshotPath = $this->snapshotPath;

        if (!$snapshotPath) {
            $snapshotPath = sys_get_temp_dir() . '/rubixml-snapshot-' . uniqid() . '.dat';
        }

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

            if ($loss <= 0.0) {
                break;
            }

            $evalThisStep = $epoch % $this->evalInterval === 0 && !$testing->empty();

            if ($evalThisStep) {
                $predictions = $this->predict($testing);

                $score = $this->metric->score($predictions, $testing->labels());

                $this->scores[$epoch] = $score;
            }

            if ($this->logger) {
                $message = "Epoch: $epoch, {$this->costFn}: $loss";

                if ($evalThisStep) {
                    $message .= ", {$this->metric}: $score";
                }

                $this->logger->info($message);
            }

            if ($evalThisStep) {
                if ($score >= $maxScore) {
                    break;
                }

                if ($score > $bestScore) {
                    $bestScore = $score;
                    $bestEpoch = $epoch;

                    if ($snapshot) {
                        $snapshot->destroy();
                    }

                    $snapshot = Snapshot::take($this->network, $snapshotPath);

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

        if ($snapshot) {
            if (end($this->scores) < $bestScore or is_nan($loss)) {
                $snapshot->restore();

                if ($this->logger) {
                    $this->logger->info("Model state restored to epoch $bestEpoch");
                }
            }

            $snapshot->destroy();
        }

        if ($this->logger) {
            $this->logger->info('Training complete');
        }
    }

    /**
     * Make predictions from a dataset.
     *
     * @param Dataset $dataset
     * @return list<string|int>
     */
    public function predict(Dataset $dataset) : array
    {
        return array_map('Rubix\ML\argmax', $this->proba($dataset));
    }

    /**
     * Estimate the joint probabilities for each possible outcome.
     *
     * @param Dataset $dataset
     * @throws RuntimeException
     * @return list<array<string|int,float>>
     */
    public function proba(Dataset $dataset) : array
    {
        if (!$this->network or !$this->classes) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        DatasetHasDimensionality::with($dataset, $this->network->input()->width())->check();

        [$classA, $classB] = $this->classes;

        $activations = $this->network->infer($dataset);

        $activations = array_column($activations->asArray(), 0);

        $probabilities = [];

        foreach ($activations as $activation) {
            $probabilities[] = [
                $classA => 1.0 - $activation,
                $classB => $activation,
            ];
        }

        return $probabilities;
    }

    /**
     * Return the importance scores of each feature column of the training set.
     *
     * @throws RuntimeException
     * @return float[]
     */
    public function featureImportances() : array
    {
        if (!$this->network) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        $layer = current($this->network->hidden());

        if (!$layer instanceof Dense) {
            throw new RuntimeException('Weight layer not found.');
        }

        $weights = $layer->weights()->abs()->asArray();

        return $weights[0] ?? [];
    }

    /**
     * Return an associative array containing the data used to serialize the object.
     *
     * @return mixed[]
     */
    public function __serialize() : array
    {
        $properties = get_object_vars($this);

        unset(
            $properties['losses'],
            $properties['scores'],
            $properties['logger'],
            $properties['snapshotPath']
        );

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
        return 'Logistic Regression (' . Params::stringify($this->params()) . ')';
    }
}
