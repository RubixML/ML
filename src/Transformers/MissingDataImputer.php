<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\Datasets\DataFrame;
use Rubix\ML\Other\Strategies\Continuous;
use Rubix\ML\Other\Strategies\BlurryMean;
use Rubix\ML\Other\Strategies\Categorical;
use Rubix\ML\Other\Strategies\PopularityContest;
use InvalidArgumentException;
use RuntimeException;

/**
 * Missing Data Imputer
 *
 * In the real world, it is common to have data with missing values here and
 * there. The Missing Data Imputer replaces missing value placeholders with a
 * guess based on a given imputer Strategy.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class MissingDataImputer implements Transformer
{
    /**
     * The placeholder of a missing value.
     *
     * @var mixed
     */
    protected $placeholder;

    /**
     * The imputer to use when imputing continuous values.
     *
     * @var \Rubix\ML\Other\Strategies\Continuous
     */
    protected $continuous;

    /**
     * The imputer to use when imputing categorical values.
     *
     * @var \Rubix\ML\Other\Strategies\Categorical
     */
    protected $categorical;

    /**
     * The fitted data imputers.
     *
     * @var array|null
     */
    protected $imputers;

    /**
     * @param  mixed  $placeholder
     * @param  \Rubix\ML\Other\Strategies\Continuous|null  $continuous
     * @param  \Rubix\ML\Other\Strategies\Categorical|null  $categorical
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct($placeholder = '?', Continuous $continuous = null, Categorical $categorical = null)
    {
        if (!is_numeric($placeholder) and !is_string($placeholder)) {
            throw new InvalidArgumentException('Placeholder must be a string or'
                . ' numeric type, ' . gettype($placeholder) . ' found.');
        }

        if (is_null($continuous)) {
            $continuous = new BlurryMean();
        }

        if (is_null($categorical)) {
            $categorical = new PopularityContest();
        }

        $this->placeholder = $placeholder;
        $this->continuous = $continuous;
        $this->categorical = $categorical;
    }

    /**
     * Fit the transformer to the incoming data frame.
     *
     * @param  \Rubix\ML\Datasets\DataFrame  $dataframe
     * @return void
     */
    public function fit(DataFrame $dataframe) : void
    {
        $this->imputers = [];

        foreach ($dataframe->types() as $column => $type) {
            if ($type === DataFrame::CATEGORICAL) {
                $imputer = clone $this->categorical;
            } else {
                $imputer = clone $this->continuous;
            }

            $values = array_filter($dataframe->column($column), function ($value) {
                return $value !== $this->placeholder;
            });

            $imputer->fit($values);

            $this->imputers[$column] = $imputer;
        }
    }

    /**
     * Apply the transformation to the samples in the data frame.
     *
     * @param  array  $samples
     * @throws \RuntimeException
     * @return void
     */
    public function transform(array &$samples) : void
    {
        if (is_null($this->imputers)) {
            throw new RuntimeException('Transformer has not been fitted.');
        }

        foreach ($samples as $row => &$sample) {
            foreach ($sample as $column => &$feature) {
                if ($feature === $this->placeholder) {
                    $strategy = $this->imputers[$column];

                    $feature = $strategy->guess();
                }
            }
        }
    }
}
