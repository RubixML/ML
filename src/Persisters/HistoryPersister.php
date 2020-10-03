<?php

namespace Rubix\ML\Persisters;

use Rubix\ML\Encoding;
use Rubix\ML\Other\Helpers\Params;
use Rubix\ML\Persistable;
use Rubix\ML\Persisters\Serializers\Native;
use Rubix\ML\Persisters\Serializers\Serializer;
use Rubix\ML\Storage\Datastore;
use Rubix\ML\Storage\Exceptions\RuntimeException;
use Rubix\ML\Storage\Streams\Stream;

/**
 * HistoryPersister
 *
 * HistoryPersister provides the logic to keep the save history of a `Persistable` object.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 * @author      Chris Simpson
 */
abstract class HistoryPersister implements Persister
{
    protected const HISTORY_EXT = 'old';

    /**
     * @var string
     */
    protected $location;

    /**
     * @var Datastore
     */
    protected $storage;

    /**
     * @var bool
     */
    protected $history;

    /**
     * @var Serializer
     */
    protected $serializer;

    /**
     * @param string $location
     * @param \Rubix\ML\Storage\Datastore $datastore
     * @param bool $history
     * @param \Rubix\ML\Persisters\Serializers\Serializer|null $serializer
     */
    public function __construct(string $location, Datastore $datastore, bool $history = true, ?Serializer $serializer = null)
    {
        $this->location = $location;
        $this->storage = $datastore;
        $this->history = $history;
        $this->serializer = $serializer ?? new Native();
    }

    /**
     * Save the persistable object.
     *
     * @param \Rubix\ML\Persistable $persistable
     * @throws \Rubix\ML\Storage\Exceptions\StorageException
     */
    public function save(Persistable $persistable) : void
    {
        if ($this->history and $this->storage->exists($this->location)) {
            $timestamp = (string) time();

            $filename = "{$this->location}-$timestamp." . self::HISTORY_EXT;

            $num = 0;

            while ($this->storage->exists($filename)) {
                $filename = "{$this->location}-$timestamp-" . ++$num . '.' . self::HISTORY_EXT;
            }

            $this->storage->move($this->location, $filename);
        }

        $encoding = $this->serializer->serialize($persistable);

        if ($encoding->bytes() === 0) {
            throw new RuntimeException("Cannot save empty encoding to {$this->location}");
        }

        $this->storage->write($this->location, $encoding->data());
    }

    /**
     * Load the last saved persistable instance.
     *
     * @throws \Rubix\ML\Storage\Exceptions\StorageException
     * @return \Rubix\ML\Persistable
     */
    public function load() : Persistable
    {
        if (!$this->storage->exists($this->location)) {
            throw new RuntimeException("Target: '{$this->location}' does not exist.");
        }

        $stream = $this->storage->read($this->location, Stream::READ_ONLY);
        $encoding = new Encoding($stream->contents());

        if ($encoding->bytes() === 0) {
            throw new RuntimeException("Target: '{$this->location}' does not"
                . ' contain any data.');
        }

        return $this->serializer->unserialize($encoding);
    }

    /**
     * @return array<mixed>
     */
    protected function params() : array
    {
        return [
            'path' => $this->location,
            'history' => $this->history,
            'serializer' => $this->serializer,
        ];
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return Params::shortName(get_called_class()) .
            ' (' . Params::stringify($this->params()) . ')';
    }
}
