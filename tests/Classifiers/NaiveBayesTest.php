<?php

namespace Rubix\Tests\Classifiers;

use Rubix\ML\Online;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Probabilistic;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Classifiers\Multiclass;
use Rubix\ML\Classifiers\Classifier;
use Rubix\ML\Classifiers\NaiveBayes;
use PHPUnit\Framework\TestCase;
use InvalidArgumentException;

class NaiveBayesTest extends TestCase
{
    protected $estimator;

    protected $training;

    protected $testing;

    public function setUp()
    {
        $samples = [
            ['nice', 'furry', 'friendly', 'happy'],
            ['mean', 'furry', 'loner', 'sad'],
            ['nice', 'rough', 'friendly', 'happy'],
            ['mean', 'rough', 'friendly', 'sad'],
            ['mean', 'rough', 'loner', 'happy'],
            ['nice', 'rough', 'loner', 'sad'],
            ['nice', 'furry', 'friendly', 'sad'],
            ['nice', 'furry', 'friendly', 'happy'],
            ['mean', 'rough', 'friendly', 'sad'],
            ['nice', 'rough', 'loner', 'sad'],
            ['mean', 'furry', 'loner', 'sad'],
            ['nice', 'furry', 'loner', 'happy'],
            ['mean', 'rough', 'loner', 'happy'],
            ['nice', 'furry', 'friendly', 'sad'],
            ['nice', 'furry', 'loner', 'sad'],
            ['mean', 'rough', 'friendly', 'sad'],
            ['nice', 'rough', 'friendly', 'happy'],
        ];

        $labels = [
            'not monster', 'monster', 'not monster', 'monster', 'monster',
            'not monster', 'not monster', 'not monster', 'monster', 'monster',
            'monster', 'not monster', 'monster', 'not monster', 'not monster',
            'monster', 'not monster',
        ];

        $this->training = new Labeled($samples, $labels);

        $this->testing = new Labeled([
            ['nice', 'rough', 'friendly', 'happy'],
            ['mean', 'furry', 'loner', 'sad'],
        ], [
            'not monster', 'monster',
        ]);

        $this->estimator = new NaiveBayes();
    }

    public function test_build_classifier()
    {
        $this->assertInstanceOf(NaiveBayes::class, $this->estimator);
        $this->assertInstanceOf(Multiclass::class, $this->estimator);
        $this->assertInstanceOf(Classifier::class, $this->estimator);
        $this->assertInstanceOf(Online::class, $this->estimator);
        $this->assertInstanceOf(Probabilistic::class, $this->estimator);
        $this->assertInstanceOf(Estimator::class, $this->estimator);
        $this->assertInstanceOf(Persistable::class, $this->estimator);
    }

    public function test_make_prediction()
    {
        $this->training->randomize();

        $this->estimator->train($this->training);

        $predictions = $this->estimator->predict($this->testing);

        $this->assertEquals($this->testing->label(0), $predictions[0]);
        $this->assertEquals($this->testing->label(1), $predictions[1]);
    }

    public function test_predict_proba()
    {
        $this->training->randomize();

        $this->estimator->train($this->training);

        $probabilities = $this->estimator->proba($this->testing);

        $this->assertGreaterThanOrEqual(0.5, $probabilities[0]['not monster']);
        $this->assertLessThan(0.5, $probabilities[0]['monster']);
        $this->assertLessThan(0.5, $probabilities[1]['not monster']);
        $this->assertGreaterThanOrEqual(0.5, $probabilities[1]['monster']);
    }

    public function test_partial_train()
    {
        $folds = $this->training->randomize()->stratifiedFold(2);

        $this->estimator->train($folds[0]);

        $this->estimator->partial($folds[1]);

        $predictions = $this->estimator->predict($this->testing);

        $this->assertEquals($this->testing->label(0), $predictions[0]);
        $this->assertEquals($this->testing->label(1), $predictions[1]);
    }

    public function test_train_with_unlabeled()
    {
        $dataset = new Unlabeled([['bad']]);

        $this->expectException(InvalidArgumentException::class);

        $this->estimator->train($dataset);
    }
}
