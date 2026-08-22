<?php

namespace Rubix\ML\Benchmarks\Classifiers;

use Rubix\ML\Backends\Backend;
use Rubix\ML\Classifiers\OneVsRest;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Classifiers\LogisticRegression;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\NeuralNet\Optimizers\Stochastic;
use Rubix\ML\Datasets\Generators\Agglomerate;
use Rubix\ML\Tests\DataProvider\BackendProviderTrait;

/**
 * @Groups({"Classifiers"})
 * @BeforeMethods({"setUp"})
 */
class OneVsRestBench
{
    use BackendProviderTrait;

    protected const int TRAINING_SIZE = 10000;

    protected const int TESTING_SIZE = 10000;

    protected Labeled $training;

    protected Labeled $testing;

    protected OneVsRest $estimator;

    public function setUp() : void
    {
        $generator = new Agglomerate([
            'Iris-setosa' => new Blob([5.0, 3.42, 1.46, 0.24], [0.35, 0.38, 0.17, 0.1]),
            'Iris-versicolor' => new Blob([5.94, 2.77, 4.26, 1.33], [0.51, 0.31, 0.47, 0.2]),
            'Iris-virginica' => new Blob([6.59, 2.97, 5.55, 2.03], [0.63, 0.32, 0.55, 0.27]),
        ]);

        $this->training = $generator->generate(self::TRAINING_SIZE);

        $this->testing = $generator->generate(self::TESTING_SIZE);

        $this->estimator = new OneVsRest(new LogisticRegression(64, new Stochastic(0.001)));
    }

    /**
     * @Subject
     * @Iterations(5)
     * @OutputTimeUnit("seconds", precision=3)
     * @ParamProviders("provideBackends")
     * @param array{ backend: Backend } $params
     */
    public function trainPredict(array $params) : void
    {
        $this->estimator->setBackend($params['backend']);

        $this->estimator->train($this->training);

        $this->estimator->predict($this->testing);
    }
}
