<?php

namespace Rubix\Engine\Persisters;

use InvalidArgumentException;
use RuntimeException;
use PDO;

class Sqlite implements Persister
{
    const DEFAULT_OPTIONS = [
        PDO::ATTR_ERRMODE => PDO::ERRMODE_EXCEPTION,
    ];

    /**
     * The path to the Sqlite database file.
     *
     * @var string
     */
    protected $path;

    /**
     * The Sqlite table that stores the model data.
     *
     * @var string
     */
    protected $table;

    /**
     * Should we maintain a history or overwrite the model each time?
     *
     * @var bool
     */
    protected $history;

    /**
     * The PDO connection.
     *
     * @var \PDO
     */
    protected $connection;

    /**
     * @param  string  $path
     * @param  string  $table
     * @param  bool  $history
     * @param  array  $options
     * @throws \InvalidArgumentException
     * @throws \RuntimeException
     * @return void
     */
    public function __construct(string $path, string $table, bool $history = true, array $options = [])
    {
        if (!is_writable($path)) {
            throw new InvalidArgumentException('Database file does not exist, or is not writeable.');
        }

        if (!extension_loaded('pdo_sqlite')) {
            throw new RuntimeException('The Sqlite PHP extension is not loaded.');
        }

        $this->connection = new PDO("sqlite:{$path}", null, null, $options);
        $this->table = addslashes($table);
        $this->history = $history;

        $this->connection->query("CREATE TABLE IF NOT EXISTS {$this->table} (model BLOB, created_at INTEGER);");
    }

    /**
     * @param  \Rubix\Engine\Persisters\Persistable  $model
     * @throws \RuntimeException
     * @return bool
     */
    public function save(Persistable $model) : bool
    {
        $model = serialize($model);

        if ((strlen($model) * 1e-9) > 2.1) {
            throw new RuntimeException('Sqlite cannot handle models larger than 2.1 gigabytes.');
        }

        if ($this->history === false) {
            $this->truncate();
        }

        return $this->connection->prepare("INSERT INTO {$this->table} (model, created_at) VALUES (?, ?);")
            ->execute([$model, time()]);
    }

    /**
     * @throws \RuntimeException
     * @return \Rubix\Engine\Persistable|null
     */
    public function restore() : ?Persistable
    {
        $result = $this->connection->query("SELECT * FROM {$this->table} ORDER BY created_at DESC LIMIT 1;")
            ->fetch();

        if ($result === false) {
            throw new RuntimeException('Could not load model from the database.');
        }

        return unserialize($result['model']) ?: null;
    }

    /**
     * Truncte the database table. i.e. empty all the data.
     *
     * @return void
     */
    public function truncate() : void
    {
        $this->connection->query("DELETE FROM {$this->table};");
    }
}
