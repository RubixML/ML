<?php

namespace Rubix\ML;

use Rubix\ML\Probabilistic;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Other\Helpers\Stats;
use Rubix\ML\Other\Helpers\Params;
use Rubix\ML\Other\Functions\Argmax;
use Rubix\ML\Other\Traits\LoggerAware;
use InvalidArgumentException;

/**
 * Model Orchestra
 *
 * A Model Orchestra is a stacked model ensemble comprised of an orchestra
 * of estimators (Classifiers or Regressors) and a conductor estimator. The
 * role of the conductor is to learn the influence scores of each estimator
 * in the orchestra while using their predictions as inputs to make a final
 * weighted prediction.
 *
 * > **Note**: The features that each estimator passes on to the conductor
 * may vary depending on the type of estimator. For example, a Probabilistic
 * classifier will pass class probability scores while a regressor will pass
 * on a single real value. If a datatype is not compatible with the
 * conducting estimator, then wrap it in a Pipeline and use a transformer
 * such as One Hot Encoder or Interval Discretizer.
 *
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class ModelOrchestra implements Learner, Persistable, Verbose
{
    use LoggerAware;

    /**
     * The estimators that comprise the orchestra.
     *
     * @var array
     */
    protected $orchestra = [
        //
    ];

    /**
     * The learner responsible for making the final prediction given
     * the predictions from the orchestra as input features.
     *
     * @var \Rubix\ML\Learner
     */
    protected $conductor;

    /**
     * The ratio of training samples to train each orchestra member
     * on with the rest being used to train the conductor.
     *
     * @var float
     */
    protected $ratio;

    /**
     * The data types that the orchestra is compatible with.
     *
     * @var int[]
     */
    protected $compatibility;

    /**
     * @param  array  $orchestra
     * @param  \Rubix\ML\Learner  $conductor
     * @param  float  $ratio
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(array $orchestra, Learner $conductor, float $ratio = 0.8)
    {
        foreach ($orchestra as $estimator) {
            $type = $estimator->type();

            if ($type !== self::CLASSIFIER and $type !== self::REGRESSOR) {
                throw new InvalidArgumentException('This meta estimator only'
                    . ' supports classifiers, and regressors, '
                    . self::TYPES[$type] . ' given.');
            }

            if ($type !== reset($orchestra)->type()) {
                throw new InvalidArgumentException('Each estimator must be of'
                    . ' the same type.');
            }
        }

        $compatibility = array_intersect(...array_map(function ($estimator) {
            return $estimator->compatibility();
        }, $orchestra));

        if (count($compatibility) < 1) {
            throw new InvalidArgumentException('Orchestra must only'
                . ' contain estimators that share at least 1 data type'
                . ' they are compatible with.');
        }

        if ($conductor->type() !== reset($orchestra)->type()) {
            throw new InvalidArgumentException('The conductor must be the same'
                . ' type as the rest of the orchestra, '
                . self::TYPES[$conductor->type()] . ' found.');
        }

        if ($ratio < 0.01 or $ratio > 0.99) {
            throw new InvalidArgumentException('Ratio must be between'
                . " 0.01 and 0.99, $ratio given.");
        }

        $this->orchestra = $orchestra;
        $this->conductor = $conductor;
        $this->ratio = $ratio;
        $this->compatibility = $compatibility;
    }

    /**
     * Return the integer encoded estimator type.
     *
     * @return int
     */
    public function type() : int
    {
        return $this->conductor->type();
    }

    /**
     * Return the data types that this estimator is compatible with.
     *
     * @return int[]
     */
    public function compatibility() : array
    {
        return $this->compatibility;
    }

    /**
     * Has the learner been trained?
     *
     * @return bool
     */
    public function trained() : bool
    {
        return $this->conductor->trained();
    }

    /**
     * Return the estimators that comprise the orchestra part of the
     * ensemble.
     *
     * @return \Rubix\ML\Estimator[]
     */
    public function orchestra() : array
    {
        return $this->orchestra;
    }

    /**
     * Return the conductor of the ensemble.
     *
     * @return \Rubix\ML\Estimator
     */
    public function conductor() : Estimator
    {
        return $this->conductor;
    }

    /**
     * Instantiate and train each base estimator in the ensemble on a bootstrap
     * training set.
     *
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

        if ($this->logger) {
            $this->logger->info('Learner initialized w/ '
            . Params::stringify([
                'orchestra' => $this->orchestra,
                'conductor' => $this->conductor,
                'ratio' => $this->ratio,
            ]));
        }

        if ($this->type() === self::CLASSIFIER) {
            [$left, $right] = $dataset->stratifiedSplit($this->ratio);
        } else {
            [$left, $right] = $dataset->randomize()->split($this->ratio);
        }

        foreach ($this->orchestra as $estimator) {
            if ($this->logger) {
                $this->logger->info('Training '
                . Params::shortName($estimator));
            }

            $estimator->train($left);
        }

        if ($right instanceof Labeled) {
            $right = $this->extract($right);
        }

        if ($this->logger) {
            $this->logger->info('Training '
            . Params::shortName($this->conductor) . ' (conductor)');
        }

        $this->conductor->train($right);

        if ($this->logger) {
            $this->logger->info('Training complete');
        }
    }

    /**
     * Make predictions from a dataset.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @throws \RuntimeException
     * @return array
     */
    public function predict(Dataset $dataset) : array
    {
        if ($dataset instanceof Labeled) {
            $dataset = $this->extract($dataset);
        }

        return $this->conductor->predict($dataset);
    }

    /**
     * Extract the features from the orchestra and return them in a
     * new dataset.
     *
     * @param  \Rubix\ML\Datasets\Labeled  $dataset
     * @return \Rubix\ML\Datasets\Labeled
     */
    protected function extract(Labeled $dataset) : Labeled
    {
        $samples = array_fill(0, $dataset->numRows(), []);

        foreach ($this->orchestra as $estimator) {
            if ($estimator instanceof Probabilistic) {
                $probabilities = $estimator->proba($dataset);

                foreach ($probabilities as $i => $dist) {
                    $features = array_values($dist);

                    $samples[$i] = array_merge($samples[$i], $features);
                }
            } else {
                $predictions = $estimator->predict($dataset);

                foreach ($predictions as $i => $feature) {
                    $samples[$i][] = $feature;
                }
            }
        }

        return Labeled::quick($samples, $dataset->labels());
    }
}
