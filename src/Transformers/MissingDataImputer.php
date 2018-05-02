<?php

namespace Rubix\Engine\Transformers;

use Rubix\Engine\Datasets\Dataset;
use Rubix\Engine\Transformers\Imputers\Imputer;
use Rubix\Engine\Transformers\Imputers\BlurryMean;
use Rubix\Engine\Transformers\Imputers\PopularityContest;
use InvalidArgumentException;

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
     * @var \Rubix\Engine\Transformers\Strategies\Continuous
     */
    protected $continuous;

    /**
     * The imputer to use when imputing categorical values.
     *
     * @var \Rubix\Engine\Transformers\Strategies\Categorical
     */
    protected $categorical;

    /**
     * The type of each feature column. i.e. categorical or continuous.
     *
     * @var array
     */
    protected $columnTypes = [
        //
    ];

    /**
     * The fitted data imputers.
     *
     * @var array
     */
    protected $imputers = [
        //
    ];

    /**
     * @param  mixed  $placeholder
     * @param  \Rubix\Engine\Transformers\Strategies\Continuous|null  $continuous
     * @param  \Rubix\Engine\Transformers\Strategies\Categorical|null  $categorical
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct($placeholder = '?', Continuous $continuous = null, Categorical $categorical = null)
    {
        if (!is_numeric($placeholder) && !is_string($placeholder)) {
            throw new InvalidArgumentException('Placeholder must be a string or numeric type, '
                . gettype($placeholder) . ' found.');
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
     * @param  \Rubix\Engine\Datasets\Dataset  $dataset
     * @return void
     */
    public function fit(Dataset $dataset) : void
    {
        $types = $dataset->columnTypes();

        foreach ($dataset->rotate() as $column => $values) {
            if ($types[$column] === self::CATEGORICAL) {
                $imputer = clone $this->categorical;
            } else if ($types[$column] === self::CONTINUOUS) {
                $imputer = clone $this->continuous;
            }

            $values = array_filter($values, function ($value) {
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
     * @return void
     */
    public function transform(array &$samples) : void
    {
        foreach ($samples as $row => &$sample) {
            foreach ($sample as $column => &$feature) {
                if ($feature === $this->placeholder) {
                    $feature = $this->imputers[$column]->impute();
                }
            }
        }
    }
}
