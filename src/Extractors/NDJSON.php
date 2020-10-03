<?php

namespace Rubix\ML\Extractors;

use Rubix\ML\Exceptions\RuntimeException;
use Rubix\ML\Other\Helpers\JSON;
use Rubix\ML\Storage\Exceptions\StorageException;
use Rubix\ML\Storage\LocalFilesystem;
use Rubix\ML\Storage\Reader;
use Rubix\ML\Storage\ReadProxy;
use Rubix\ML\Storage\Streams\Stream;
use Generator;

/**
 * NDJSON
 *
 * NDJSON or *Newline Delimited* JSON files contain rows of data encoded in Javascript Object
 * Notation (JSON) arrays or objects. The format is like a mix of JSON and CSV and has the
 * advantage of retaining data type information and being read into memory incrementally.
 *
 * > **Note:** Empty lines are ignored by the parser.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class NDJSON implements Extractor
{
    /**
     * @var \Rubix\ML\Storage\Streams\Stream
     */
    protected $stream;

    /**
     * @param string $location
     * @param ?Reader $storage
     *
     * @throws \Rubix\ML\Storage\Exceptions\StorageException
     */
    public function __construct(string $location, ?Reader $storage = null)
    {
        if (!$storage) {
            $storage = new LocalFilesystem();
        }

        $storage = new ReadProxy($storage);

        $this->stream = $storage->read($location, Stream::READ_ONLY);
    }

    /**
     * Clean up the any open file handles.
     */
    public function __destruct()
    {
        try {
            if ($this->stream and $this->stream->open()) {
                $this->stream->close();
            }
        } catch (StorageException $e) {
            //
        }
    }

    /**
     * Return an iterator for the records in the data table.
     *
     * @throws \Rubix\ML\Exceptions\RuntimeException
     * @throws \Rubix\ML\Storage\Exceptions\StorageException
     * @return \Generator<list<mixed>>
     */
    public function getIterator() : Generator
    {
        if ($this->stream->seekable()) {
            $this->stream->rewind();
        }

        $num = 0;

        foreach ($this->stream as $line) {
            $data = rtrim($line);
            ++$num;

            if (empty($data)) {
                continue 1;
            }

            try {
                yield JSON::decode($data);
            } catch (RuntimeException $e) {
                throw new RuntimeException(
                    "JSON Error on line: $num. (" . $e->getMessage() . ')',
                    $e->getCode(),
                    $e
                );
            }
        }
    }
}
