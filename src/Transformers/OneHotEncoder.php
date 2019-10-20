<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\DataType;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Other\Specifications\DatasetIsCompatibleWithTransformer;
use RuntimeException;

/**
 * One Hot Encoder
 *
 * The One Hot Encoder takes a feature column of categorical values and produces an n-d
 * *one-hot* representation where n is equal to the number of unique categories in that
 * column. After the transformation, a 0 in any location indicates that the category
 * represented by that column is not present in the sample whereas a 1 indicates that a
 * category is present. One hot encoding is typically used to convert categorical data to
 * continuous so that it can be used to train a learner that is only compatible with
 * continuous features.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class OneHotEncoder implements Transformer, Stateful
{
    /**
     * The set of unique possible categories of the training set.
     *
     * @var array|null
     */
    protected $categories;

    /**
     * Return the data types that this transformer is compatible with.
     *
     * @return int[]
     */
    public function compatibility() : array
    {
        return DataType::ALL;
    }

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
     * Return the categories computed during fitting indexed by feature column.
     *
     * @return array|null
     */
    public function categories() : ?array
    {
        return $this->categories ? array_map('array_flip', $this->categories) : null;
    }

    /**
     * Fit the transformer to the dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     */
    public function fit(Dataset $dataset) : void
    {
        DatasetIsCompatibleWithTransformer::check($dataset, $this);
        
        $n = $dataset->numColumns();

        $this->categories = [];

        for ($column = 0; $column < $n; $column++) {
            if ($dataset->columnType($column) === DataType::CATEGORICAL) {
                $values = $dataset->column($column);
                
                $categories = array_values(array_unique($values));

                $this->categories[$column] = array_flip($categories);
            }
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
        if ($this->categories === null) {
            throw new RuntimeException('Transformer has not been fitted.');
        }

        $templates = [];

        foreach ($this->categories as $column => $categories) {
            $templates[$column] = array_fill(0, count($categories), 0);
        }

        foreach ($samples as &$sample) {
            $vector = [];

            foreach ($this->categories as $column => $categories) {
                $category = $sample[$column];
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
