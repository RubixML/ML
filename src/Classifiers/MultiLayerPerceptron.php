<?php

namespace Rubix\Engine\Classifiers;

use Rubix\Engine\Supervised;
use Rubix\Engine\Persistable;
use Rubix\Engine\Probabilistic;
use Rubix\Engine\Datasets\Dataset;
use Rubix\Engine\Datasets\Labeled;
use Rubix\Engine\NeuralNet\Network;
use Rubix\Engine\NeuralNet\Layers\Input;
use Rubix\Engine\NeuralNet\Layers\Hidden;
use Rubix\Engine\NeuralNet\Optimizers\Adam;
use Rubix\Engine\Metrics\Validation\Accuracy;
use Rubix\Engine\NeuralNet\Layers\Multiclass;
use Rubix\Engine\NeuralNet\Optimizers\Optimizer;
use Rubix\Engine\Metrics\Validation\Classification;
use InvalidArgumentException;
use RuntimeException;

class MultiLayerPerceptron implements Supervised, Probabilistic, Classifier, Persistable
{
    /**
     * The hidden layer configuration of the neural net.
     *
     * @param array
     */
    protected $hidden;

    /**
     * The number of training samples to consider per iteration of gradient descent.
     *
     * @var int
     */
    protected $batchSize;

    /**
     * The gradient descent optimizer used to train the network.
     *
     * @var \Rubix\Engine\NeuralNet\Optimizers\Optimizer
     */
    protected $optimizer;

    /**
     * The L2 regularization parameter.
     *
     * @var float
     */
    protected $alpha;

    /**
     * The minimum validation score needed to early stop training.
     *
     * @var float
     */
    protected $threshold;

    /**
     * The classification metric used to validate the performance of the model.
     *
     * @var \Rubix\Engine\Metrics\Validation\Classification
     */
    protected $metric;

    /**
     * The ratio of training samples to use for validation. i.e. the holdout
     * ratio.
     *
     * @var float
     */
    protected $ratio;

    /**
     * The training window to consider during early stop checking i.e. the last
     * n epochs.
     *
     * @var int
     */
    protected $window;

    /**
     * The maximum number of training epochs. i.e. the number of times to iterate
     * over the entire training set before the algorithm terminates.
     *
     * @var int
     */
    protected $epochs;

    /**
     * The underlying computational graph.
     *
     * @param \Rubix\Engine\NeuralNet\Network
     */
    protected $network;

    /**
     * The validation score of each epoch during training.
     *
     * @param array
     */
    protected $progress = [
        //
    ];

    /**
     * @param  array  $hidden
     * @param  int  $batchSize
     * @param  \Rubix\Engine\NeuralNet\Optimizers\Optimizer|null  $optimizer
     * @param  float  $alpha
     * @param  \Rubix\Engine\Metrics\Validation\Classification|null  $metric
     * @param  float $ratio
     * @param  int  $window
     * @param  int  $epochs
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(array $hidden = [], int $batchSize = 10, Optimizer $optimizer = null,
                    float $alpha = 1e-4, Classification $metric = null, float $ratio = 0.2,
                    int $window = 3, int $epochs = PHP_INT_MAX)
    {
        if ($batchSize < 1) {
            throw new InvalidArgumentException('Cannot have less than 1 sample'
                . ' per batch.');
        }

        if ($alpha < 0.0) {
            throw new InvalidArgumentException('L2 regularization term must'
                . ' be non-negative.');
        }

        if ($ratio < 0.01 or $ratio > 1.0) {
            throw new InvalidArgumentException('Holdout ratio must be'
                . ' between 0.01 and 1.0.');
        }

        if ($window < 1) {
            throw new InvalidArgumentException('Stopping criteria window must'
                . ' be at least 2 epochs.');
        }

        if ($epochs < 1) {
            throw new InvalidArgumentException('Estimator must train for at'
                . ' least 1 epoch.');
        }

        if (!isset($optimizer)) {
            $optimizer = new Adam();
        }

        if (!isset($metric)) {
            $metric = new Accuracy();
        }

        $this->hidden = $hidden;
        $this->batchSize = $batchSize;
        $this->optimizer = $optimizer;
        $this->alpha = $alpha;
        $this->metric = $metric;
        $this->ratio = $ratio;
        $this->window = $window;
        $this->epochs = $epochs;
    }

    /**
     * @return array
     */
    public function progress() : array
    {
        return $this->progress;
    }

    /**
     * Train the network using mini-batch gradient descent with backpropagation.
     *
     * @param  \Rubix\Engine\Datasets\Labeled  $dataset
     * @throws \InvalidArgumentException
     * @return void
     */
    public function train(Labeled $dataset) : void
    {
        if (in_array(self::CATEGORICAL, $dataset->columnTypes())) {
            throw new InvalidArgumentException('This estimator only works with'
            . ' continuous features.');
        }

        list($training, $testing) = $dataset->stratifiedSplit(1 - $this->ratio);

        $this->network = new Network(new Input($dataset->numColumns()),
            $this->hidden, new Multiclass($dataset->possibleOutcomes(),
            $this->alpha));

        foreach ($this->network->initialize()->parametric() as $layer) {
            $this->optimizer->initialize($layer);
        }

        $best = ['score' => -INF, 'snapshot' => null];

        $this->progress = [];

        for ($epoch = 1; $epoch <= $this->epochs; $epoch++) {
            foreach ($training->randomize()->batch($this->batchSize) as $batch) {
                $this->network->feed($batch->samples())
                    ->backpropagate($batch->labels());

                foreach ($this->network->parametric() as $layer) {
                    $layer->update($this->optimizer->step($layer));
                }
            }

            $score = $this->scoreEpoch($testing);

            $this->progress[$epoch] = $score;

            if ($score > $best['score']) {
                $best['score'] = $score;
                $best['snapshot'] = $this->network->read();
            }

            if ($epoch >= $this->window) {
                $window = array_slice($this->progress, -$this->window);

                $worst = $window;
                rsort($worst);

                if ($window === $worst) {
                    break 1;
                }
            }
        }

        if ($score !== $best['score']) {
            $this->network->restore($best['snapshot']);
        }
    }

    /**
     * Feed a sample through the network and make a prediction based on the highest
     * activated output neuron.
     *
     * @param  \Rubix\Engine\Datasets\Dataset  $samples
     * @return array
     */
    public function predict(Dataset $samples) : array
    {
        $predictions = [];

        foreach ($this->proba($samples) as $probabilities) {
            $best = ['probability' => -INF, 'outcome' => null];

            foreach ($probabilities as $class => $probability) {
                if ($probability > $best['probability']) {
                    $best['probability'] = $probability;
                    $best['outcome'] = $class;
                }
            }

            $predictions[] = $best['outcome'];
        }

        return $predictions;
    }

    /**
     * Output a vector of class probabilities per sample.
     *
     * @param  \Rubix\Engine\Datasets\Dataset  $samples
     * @return array
     */
    public function proba(Dataset $samples) : array
    {
        $this->network->feed($samples->samples());

        return $this->network->output()->activations();
    }

    /**
     * Score the training round with supplied classification metric.
     *
     * @param  \Rubix\Engine\Dataset\Labeled  $dataset
     * @return float
     */
    protected function scoreEpoch(Labeled $dataset) : float
    {
        $predictions = $this->predict($dataset);

        $score = $this->metric->score($predictions, $dataset->labels());

        return $score;
    }
}
