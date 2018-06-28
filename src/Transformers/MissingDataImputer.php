<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Transformers\Strategies\Continuous;
use Rubix\ML\Transformers\Strategies\BlurryMean;
use Rubix\ML\Transformers\Strategies\Categorical;
use Rubix\ML\Transformers\Strategies\PopularityContest;
use InvalidArgumentException;
use RuntimeException;

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
     * @var \Rubix\ML\Transformers\Strategies\Continuous
     */
    protected $continuous;

    /**
     * The imputer to use when imputing categorical values.
     *
     * @var \Rubix\ML\Transformers\Strategies\Categorical
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
     * @param  \Rubix\ML\Transformers\Strategies\Continuous|null  $continuous
     * @param  \Rubix\ML\Transformers\Strategies\Categorical|null  $categorical
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct($placeholder = '?', Continuous $continuous = null,
                                Categorical $categorical = null)
    {
        if (!is_numeric($placeholder) and !is_string($placeholder)) {
            throw new InvalidArgumentException('Placeholder must be a string or'
                . ' numeric type, ' . gettype($placeholder) . ' found.');
        }

        if (!isset($continuous)) {
            $continuous = new BlurryMean();
        }

        if (!isset($categorical)) {
            $categorical = new PopularityContest();
        }

        $this->placeholder = $placeholder;
        $this->continuous = $continuous;
        $this->categorical = $categorical;
    }

    /**
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @return void
     */
    public function fit(Dataset $dataset) : void
    {
        $this->imputers = [];

        foreach ($dataset->columnTypes() as $column => $type) {
            if ($type === self::CATEGORICAL) {
                $imputer = clone $this->categorical;
            } else {
                $imputer = clone $this->continuous;
            }

            $values = array_filter($dataset->column($column),
                function ($value) {
                    return $value !== $this->placeholder;
                });

            $imputer->fit($values);

            $this->imputers[$column] = $imputer;
        }
    }

    /**
     * Replace missing values within sample set with guessed values.
     *
     * @param  array  $samples
     * @throws \RuntimeException
     * @return void
     */
    public function transform(array &$samples) : void
    {
        if (!isset($this->imputers)) {
            throw new RuntimeException('Transformer has not been fitted.');
        }

        foreach ($samples as $row => &$sample) {
            foreach ($sample as $column => &$feature) {
                if ($feature === $this->placeholder) {
                    $feature = $this->imputers[$column]->guess();
                }
            }
        }
    }
}
