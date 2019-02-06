<?php

namespace Rubix\ML\Transformers;

use InvalidArgumentException;

/**
 * Stop Word Filter
 *
 * Removes user-specified words from a corpus of text.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class StopWordFilter implements Transformer
{
    /**
     * A list of stop words to filter out of each text feature.
     * 
     * @var string[]
     */
    protected $stopWords;

    /**
     * @param  array  $stopWords
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(array $stopWords = [])
    {
        $regexs = [];

        foreach ($stopWords as $word) {
            if (!is_string($word)) {
                throw new InvalidArgumentException('Stop word must be a string, '
                    . gettype($word) . ' found.');
            }

            $regexs[] = '/\b' . preg_quote($word, '/') . '\b/';
        }

        $this->stopWords = array_combine($stopWords, $regexs) ?: [];
    }

    /**
     * Transform the dataset in place.
     *
     * @param  array  $samples
     * @return void
     */
    public function transform(array &$samples) : void
    {
        foreach ($samples as &$sample) {
            foreach ($sample as &$feature) {
                if (is_string($feature)) {
                    $feature = preg_replace($this->stopWords, '', $feature);
                }
            }
        }
    }
}
