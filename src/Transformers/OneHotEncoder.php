<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Other\Helpers\DataType;
use RuntimeException;

/**
 * One Hot Encoder
 *
 * The One Hot Encoder takes a column of categorical features and produces a n-d
 * one-hot (numerical) representation where n is equal to the number of unique
 * categories in that column. A 0 indicates that a category is not present in the
 * sample whereas a 1 indicates that a category is present.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class OneHotEncoder implements Stateful
{
    /**
     * The set of unique possible categories of the training set.
     *
     * @var array|null
     */
    protected $categories;

    /**
     * Is the transformer fitted?
     *
     * @return bool
     */
    public function fitted() : bool
    {
        return isset($this->categories);
    }

    /**
     * Fit the transformer to the dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     */
    public function fit(Dataset $dataset) : void
    {
        $columns = $dataset->columnsByType(DataType::CATEGORICAL);

        $this->categories = [];

        foreach ($columns as $column => $values) {
            $categories = array_values(array_unique($values));

            $this->categories[$column] = array_flip($categories);
        }
    }

    /**
     * Transform the dataset in place.
     *
     * @param array $samples
     * @throws \RuntimeException
     */
    public function transform(array &$samples) : void
    {
        if (is_null($this->categories)) {
            throw new RuntimeException('Transformer has not been fitted.');
        }

        $templates = [];

        foreach ($this->categories as $column => $categories) {
            $templates[$column] = array_fill(0, count($categories), 0);
        }

        foreach ($samples as &$sample) {
            $vector = [];

            foreach ($this->categories as $column => $categories) {
                $category = $sample[$column] ?? null;

                $features = $templates[$column];

                if (isset($categories[$category])) {
                    $features[$categories[$category]] = 1;
                }

                $vector = array_merge($vector, $features);

                unset($sample[$column]);
            }

            $sample = array_merge($sample, $vector);
        }
    }
}
