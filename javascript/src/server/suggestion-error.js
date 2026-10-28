export class SuggestionError extends Error {
    constructor(code, message, status) {
        super(message);
        this.name = 'SuggestionError';
        this.code = code;
        this.status = status;
    }
}
