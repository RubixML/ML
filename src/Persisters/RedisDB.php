<?php

namespace Rubix\ML\Persisters;

use Rubix\ML\Persistable;
use Rubix\ML\Persisters\Serializers\Native;
use Rubix\ML\Persisters\Serializers\Serializer;
use InvalidArgumentException;
use RuntimeException;
use Redis;

/**
 * Redis DB
 *
 * Redis is a high performance in-memory key value store that can be used to
 * persist models. The persiter requires the PHP Redis extension and a properly
 * configured Redis server.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class RedisDB implements Persister
{
    /**
     * The key of the model in storage.
     *
     * @var string
     */
    protected $key;

    /**
     * The number of backups to keep.
     *
     * @var int
     */
    protected $history;

    /**
     * The connector to the Redis database.
     *
     * @var \Redis
     */
    protected $connector;

    /**
     * The serializer used to convert to and from serial format.
     *
     * @var \Rubix\ML\Persisters\Serializers\Serializer
     */
    protected $serializer;

    /**
     * @param string $key
     * @param string $host
     * @param int $port
     * @param int $db
     * @param string $password
     * @param int $history
     * @param \Rubix\ML\Persisters\Serializers\Serializer|null $serializer
     * @param float $timeout
     * @throws \InvalidArgumentException
     * @throws \RuntimeException
     */
    public function __construct(
        string $key,
        string $host = '127.0.0.1',
        int $port = 6379,
        int $db = 0,
        string $password = null,
        int $history = 2,
        ?Serializer $serializer = null,
        float $timeout = 2.5
    ) {
        if (!extension_loaded('redis')) {
            throw new RuntimeException('Redis extension is not loaded, check'
                . ' PHP configuration.');
        }

        if ($timeout <= 0.) {
            throw new InvalidArgumentException('Timeout must be greater than 0'
                . ", $timeout given.");
        }

        $connector = new Redis();

        if (!$connector->connect($host, $port, $timeout)) {
            throw new RuntimeException('Could not connect to Redis server'
                . ' at host ' . $host . ' on port ' . (string) $port . '.');
        };

        if (isset($password)) {
            if (!$connector->auth($password)) {
                throw new RuntimeException('Password is invalid.');
            }
        }

        if (!$connector->select($db)) {
            throw new RuntimeException('Could not select database number'
                . (string) $db . '.');
        };

        if ($history < 0) {
            throw new InvalidArgumentException('The number of backups'
                . " cannot be less than 0, $history given.");
        }

        if (is_null($serializer)) {
            $serializer = new Native();
        }

        $this->key = $key;
        $this->history = $history;
        $this->connector = $connector;
        $this->serializer = $serializer;
    }

    /**
     * Return an associative array of info from the Redis server.
     *
     * @return array
     */
    public function info() : array
    {
        return $this->connector->info();
    }

    /**
     * Save the persitable object.
     *
     * @param \Rubix\ML\Persistable $persistable
     * @throws \RuntimeException
     */
    public function save(Persistable $persistable) : void
    {
        $data = $this->serializer->serialize($persistable);

        $success = $length = $this->connector->rPush($this->key, $data);

        if (!$success) {
            throw new RuntimeException('There was an error saving the'
                . ' model to the database.');
        };

        $remove = $length - ($this->history + 1);

        for ($i = 0; $i < $remove; $i++) {
            if (!$this->connector->lPop($this->key)) {
                throw new RuntimeException('There was an error'
                    . ' deleting an old backup from the database.');
            };
        }
    }

    /**
     * Load the last model that was saved.
     *
     * @throws \RuntimeException
     * @return \Rubix\ML\Persistable
     */
    public function load() : Persistable
    {
        $data = $this->connector->lGet($this->key, -1) ?: '';

        $persistable = $this->serializer->unserialize($data);

        if (!$persistable instanceof Persistable) {
            throw new RuntimeException('Model could not be reconstituted.');
        }

        return $persistable;
    }
}
