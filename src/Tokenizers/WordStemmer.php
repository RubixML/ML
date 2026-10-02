<?php

namespace Rubix\ML\Tokenizers;

use Rubix\ML\Helpers\Params;

/**
 * Word Stemmer
 *
 * Word Stemmer reduces inflected and derived words to their root form using the Snowball method. For example,
 * the sentence "Majority voting is likely foolish" might stem to "Major vote is like foolish."
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class WordStemmer extends Word
{
    /**
     * The word stemmer.
     *
     * @var \Rubix\ML\Tokenizers\Stemmers\Stemmer
     */
    protected $stemmer;

    /**
     * @param string $language
     */
    public function __construct(Stemmer $stemmer)
    {
        $this->stemmer = $stemmer;
    }

    /**
     * Tokenize a block of text.
     *
     * @param string $string
     * @return string[]
     */
    public function tokenize(string $string) : array
    {
        return array_map([$this->stemmer, 'stem'], parent::tokenize($string));
    }

    /**
     * Return the string representation of the object.
     *
     * @internal
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Word Stemmer (language: ' . Params::shortName(get_class($this->stemmer)) . ')';
    }
}